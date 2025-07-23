import discord
import json
import os
import re
import math
import asyncio
import logging
import aiohttp
from typing import Any, Dict, List, Optional, Union, Tuple
from discord.ext import commands, tasks
from discord import app_commands
from rapidfuzz import fuzz, process
import aiofiles

# ---------------------------------
# Logging Setup
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.FileHandler("donation_manager.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DonationManagerCog")

# ---------------------------------
# Storage & Configuration
# ---------------------------------
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def storage_path(filename: str) -> str:
    """Helper function to get the full path for a storage file."""
    return os.path.join(STORAGE_DIR, filename)

# --- File Paths ---
CONFIG_FILE = storage_path("donation_config.json")
USER_DONATIONS_FILE = storage_path("user_donations.json")
ITEM_VALUES_FILE = storage_path("item_values.json")

# --- Constants ---
DANK_MEMER_ID = 270904126974590976
FUZZY_MATCH_THRESHOLD = 85
ITEM_API_URL = "https://api.gwapes.com/items"
LEADERBOARD_PER_PAGE = 10

# --- Regex for parsing Dank Memer messages ---
ITEM_DONATION_REGEX = re.compile(r"(?:successfully\s)?donated\s+([\d,]+)x?\s+([a-zA-Z\s'.☭]+)", re.IGNORECASE)
COIN_DONATION_REGEX = re.compile(r"successfully\sdonated\s*(?:⏣\s*)?([\d,\.kmbt]+)$", re.IGNORECASE)


# ---------------------------------
# Asynchronous File I/O
# ---------------------------------
file_lock = asyncio.Lock()

async def load_json_async(path: str, default: Any = None) -> Any:
    """Asynchronously loads a JSON file with retry logic."""
    async with file_lock:
        if not os.path.exists(path):
            return default if default is not None else {}
        for _ in range(3):
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    if not content:
                        return default if default is not None else {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading {path}: {e}. Retrying...")
                await asyncio.sleep(0.2)
        logger.critical(f"Failed to load JSON from {path} after multiple retries.")
        return default if default is not None else {}

async def save_json_async(path: str, data: Any) -> None:
    """Asynchronously saves data to a JSON file using a temporary file for safety."""
    async with file_lock:
        tmp_path = path + ".tmp"
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=4))
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Error saving data to {path}: {e}")

# ---------------------------------
# Data Parsing & Text Extraction
# ---------------------------------
def parse_dank_memer_amount(raw_str: str) -> int:
    """Converts a string like '2.1m', '2100k', or '1b' into an integer."""
    val = raw_str.lower().replace(",", "").strip()
    multiplier = 1
    if val.endswith('t'):
        multiplier = 1_000_000_000_000
        val = val[:-1]
    elif val.endswith('b'):
        multiplier = 1_000_000_000
        val = val[:-1]
    elif val.endswith('m'):
        multiplier = 1_000_000
        val = val[:-1]
    elif val.endswith('k'):
        multiplier = 1_000
        val = val[:-1]
    try:
        base = float(val)
        return int(base * multiplier)
    except ValueError:
        logger.error(f"Could not parse amount: {raw_str}")
        return 0

def normalize_text(text: str) -> str:
    """Prepares text for fuzzy matching by cleaning and simplifying it."""
    text = re.sub(r"<a?:\w+:\d+>", "", text)
    text = re.sub(r"[*_`~]", "", text)
    return text.lower().strip()

# --- Standalone Permission Check ---
async def is_admin_or_staff(interaction: discord.Interaction) -> bool:
    """Checks if a user has admin perms or a configured staff role."""
    cog = interaction.client.get_cog("Donation Manager")
    if not cog:
        logger.error("Could not find DonationManagerCog for permission check.")
        return False

    if interaction.user.guild_permissions.manage_guild:
        return True
    
    guild_id = str(interaction.guild_id)
    guild_config = cog.guild_configs.get(guild_id, {})
    staff_role_ids = guild_config.get("staff_role_ids", [])
    
    if not staff_role_ids:
        return False
        
    user_role_ids = {role.id for role in interaction.user.roles}
    for staff_id in staff_role_ids:
        if staff_id in user_role_ids:
            return True
            
    return False

# ---------------------------------
# Leaderboard View
# ---------------------------------
class LeaderboardView(discord.ui.View):
    def __init__(self, bot, sorted_donators):
        super().__init__(timeout=180)
        self.bot = bot
        self.sorted_donators = sorted_donators
        self.current_page = 0
        self.total_pages = math.ceil(len(self.sorted_donators) / LEADERBOARD_PER_PAGE)
        self.update_buttons()

    def update_buttons(self):
        self.children[0].disabled = self.current_page == 0
        self.children[1].disabled = self.current_page >= self.total_pages - 1

    async def get_page_embed(self) -> discord.Embed:
        start_index = self.current_page * LEADERBOARD_PER_PAGE
        end_index = start_index + LEADERBOARD_PER_PAGE
        page_donators = self.sorted_donators[start_index:end_index]

        embed = discord.Embed(title="Donation Leaderboard", color=discord.Color.gold())
        description = []
        for i, (user_id, data) in enumerate(page_donators, start=start_index + 1):
            try:
                user = await self.bot.fetch_user(int(user_id))
                user_mention = user.mention
            except discord.NotFound:
                user_mention = f"Unknown User (ID: {user_id})"
            total_value = data.get('total_donation_value', 0)
            description.append(f"**{i}.** {user_mention} - `⏣ {total_value:,}`")
        
        embed.description = "\n".join(description) if description else "It's lonely here..."
        embed.set_footer(text=f"Page {self.current_page + 1}/{self.total_pages}")
        return embed

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.grey)
    async def previous_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page -= 1
        self.update_buttons()
        embed = await self.get_page_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.grey)
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page += 1
        self.update_buttons()
        embed = await self.get_page_embed()
        await interaction.response.edit_message(embed=embed, view=self)

# ---------------------------------
# Main Donation Manager Cog
# ---------------------------------
class DonationManagerCog(commands.Cog, name="Donation Manager"):
    """Detects, tracks, and manages donations from Dank Memer."""

    dono = app_commands.Group(name="dono", description="Main command for donation tracking.")
    dono_staff = app_commands.Group(name="staff", parent=dono, description="Manage staff roles for the bot.")

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.item_data: Dict[str, Dict[str, Any]] = {}
        self.guild_configs: Dict[str, Dict[str, Any]] = {}
        self.user_donations_cache: Dict[str, Dict[str, Any]] = {}
        self.session = aiohttp.ClientSession()
        
        self.update_item_values_task.start()
        logger.info("DonationManagerCog initialized.")

    async def cog_load(self):
        """Asynchronous setup called when the cog is loaded."""
        await self._load_all_data()
        await self._update_item_values_from_api()

    def cog_unload(self):
        """Cleanup when the cog is unloaded."""
        self.update_item_values_task.cancel()
        
        asyncio.create_task(save_json_async(USER_DONATIONS_FILE, self.user_donations_cache))
        asyncio.create_task(self.session.close())
        logger.info("DonationManagerCog unloaded and cache saved.")

    # --- Data Loading and Caching ---
    async def _load_all_data(self):
        """Loads all necessary data from files into memory."""
        self.guild_configs = await load_json_async(CONFIG_FILE, {})
        self.user_donations_cache = await load_json_async(USER_DONATIONS_FILE, {})
        await self._load_item_data()
        logger.info(f"Loaded {len(self.guild_configs)} guild configs and {len(self.user_donations_cache)} users from files.")

    async def _load_item_data(self):
        """Loads or reloads item data from the JSON file into the cache."""
        raw_items = await load_json_async(ITEM_VALUES_FILE, [])
        if not raw_items:
            logger.warning(f"{ITEM_VALUES_FILE} is empty or not found. Item donations will not work until data is fetched.")
            self.item_data = {}
        else:
            self.item_data = {normalize_text(item['name']): item for item in raw_items}
        logger.info(f"Loaded {len(self.item_data)} items into cache.")

    # --- Background Tasks ---
    @tasks.loop(minutes=30)
    async def update_item_values_task(self):
        """Periodically fetches the latest item values from the API."""
        logger.info("Scheduled task: updating item values from API...")
        await self._update_item_values_from_api()

    async def _update_item_values_from_api(self):
        """Fetches item data from the API, saves it, and reloads the cache."""
        logger.info(f"Attempting to fetch item data from {ITEM_API_URL}")
        try:
            async with self.session.get(ITEM_API_URL) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success") and "body" in data:
                        item_list = data["body"]
                        await save_json_async(ITEM_VALUES_FILE, item_list)
                        logger.info(f"Successfully saved {len(item_list)} items to {ITEM_VALUES_FILE}.")
                        await self._load_item_data()
                    else:
                        logger.error(f"API call successful, but response format was unexpected: {data}")
                else:
                    logger.error(f"Failed to fetch item data. API returned status: {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"An aiohttp client error occurred while fetching item data: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during item data update: {e}", exc_info=True)

    async def _save_guild_configs(self):
        """Saves the guild configurations to the file."""
        await save_json_async(CONFIG_FILE, self.guild_configs)
        
    # --- Autocomplete Function ---
    async def item_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        """Provides autocomplete suggestions for item names."""
        choices = [
            app_commands.Choice(name=item['name'], value=item['name'])
            for item in self.item_data.values() if current.lower() in item['name'].lower()
        ]
        return choices[:25]

    # --- Admin/Staff Commands ---
    @dono.command(name="track", description="Enable donation tracking in a channel.")
    @app_commands.describe(channel="The channel to track donations in.")
    @app_commands.check(is_admin_or_staff)
    async def track_channel(self, interaction: discord.Interaction, channel: discord.TextChannel):
        guild_id = str(interaction.guild_id)

        if guild_id not in self.guild_configs:
            self.guild_configs[guild_id] = {"tracked_channel_ids": [], "donation_ranks": [], "log_channel_id": None, "staff_role_ids": []}
        
        tracked_channels = self.guild_configs[guild_id].setdefault("tracked_channel_ids", [])

        if channel.id in tracked_channels:
            await interaction.response.send_message(f"{channel.mention} is already being tracked.", ephemeral=True)
            return

        tracked_channels.append(channel.id)
        await self._save_guild_configs()
        await interaction.response.send_message(f"✅ Donation tracking is now **enabled** in {channel.mention}.", ephemeral=True)
        logger.info(f"Donation tracking enabled for channel {channel.id} in guild {guild_id} by {interaction.user}.")

    @dono.command(name="untrack", description="Disable donation tracking in a channel.")
    @app_commands.describe(channel="The channel to stop tracking donations in.")
    @app_commands.check(is_admin_or_staff)
    async def untrack_channel(self, interaction: discord.Interaction, channel: discord.TextChannel):
        guild_id = str(interaction.guild_id)

        if guild_id not in self.guild_configs or channel.id not in self.guild_configs[guild_id].get("tracked_channel_ids", []):
            await interaction.response.send_message(f"{channel.mention} is not being tracked.", ephemeral=True)
            return

        self.guild_configs[guild_id]["tracked_channel_ids"].remove(channel.id)
        await self._save_guild_configs()
        await interaction.response.send_message(f"❌ Donation tracking is now **disabled** in {channel.mention}.", ephemeral=True)
        logger.info(f"Donation tracking disabled for channel {channel.id} in guild {guild_id} by {interaction.user}.")

    @dono.command(name="log", description="Set, update, or disable the donation log channel.")
    @app_commands.describe(channel="The channel for logs. Leave blank to disable logging.")
    @app_commands.check(is_admin_or_staff)
    async def log_channel_toggle(self, interaction: discord.Interaction, channel: Optional[discord.TextChannel] = None):
        guild_id = str(interaction.guild_id)
        if guild_id not in self.guild_configs:
            self.guild_configs[guild_id] = {"tracked_channel_ids": [], "donation_ranks": [], "log_channel_id": None, "staff_role_ids": []}
            
        if channel:
            self.guild_configs[guild_id]["log_channel_id"] = channel.id
            await self._save_guild_configs()
            await interaction.response.send_message(f"✅ All donation logs will now be sent to {channel.mention}.", ephemeral=True)
            logger.info(f"Log channel set to {channel.id} in guild {guild_id} by {interaction.user}.")
        else:
            self.guild_configs[guild_id]["log_channel_id"] = None
            await self._save_guild_configs()
            await interaction.response.send_message("✅ Donation logging has been disabled.", ephemeral=True)
            logger.info(f"Log channel unset in guild {guild_id} by {interaction.user}.")

    @dono_staff.command(name="add-role", description="Add a role that can use the bot's admin commands.")
    @app_commands.describe(role="The role to designate as staff.")
    @app_commands.check(is_admin_or_staff)
    async def add_staff_role(self, interaction: discord.Interaction, role: discord.Role):
        guild_id = str(interaction.guild_id)
        if guild_id not in self.guild_configs:
            self.guild_configs[guild_id] = {"tracked_channel_ids": [], "donation_ranks": [], "log_channel_id": None, "staff_role_ids": []}
        
        staff_roles = self.guild_configs[guild_id].setdefault("staff_role_ids", [])
        if role.id in staff_roles:
            await interaction.response.send_message(f"{role.mention} is already a staff role.", ephemeral=True)
            return
            
        staff_roles.append(role.id)
        await self._save_guild_configs()
        await interaction.response.send_message(f"✅ {role.mention} has been added as a staff role.", ephemeral=True)
        logger.info(f"Staff role {role.id} added in guild {guild_id} by {interaction.user}.")

    @dono_staff.command(name="remove-role", description="Remove a staff role.")
    @app_commands.describe(role="The staff role to remove.")
    @app_commands.check(is_admin_or_staff)
    async def remove_staff_role(self, interaction: discord.Interaction, role: discord.Role):
        guild_id = str(interaction.guild_id)
        staff_roles = self.guild_configs.get(guild_id, {}).get("staff_role_ids", [])
        
        if role.id not in staff_roles:
            await interaction.response.send_message(f"{role.mention} is not a staff role.", ephemeral=True)
            return
            
        staff_roles.remove(role.id)
        await self._save_guild_configs()
        await interaction.response.send_message(f"✅ {role.mention} has been removed from the staff roles.", ephemeral=True)
        logger.info(f"Staff role {role.id} removed in guild {guild_id} by {interaction.user}.")

    @dono.command(name="add", description="Manually add a donation for a user.")
    @app_commands.autocomplete(item=item_autocomplete)
    @app_commands.describe(user="The user to add the donation to.", quantity="The amount of coins (e.g., 10m) or number of items.", item="The name of the item (optional, for item donations).")
    @app_commands.check(is_admin_or_staff)
    async def add_donation(self, interaction: discord.Interaction, user: discord.User, quantity: str, item: Optional[str] = None):
        await interaction.response.defer() # Public response

        value, details, donation_type, error = self._calculate_donation_value(quantity, item)
        if error:
            await interaction.followup.send(error, ephemeral=True)
            return

        details += f" (Manually Added by {interaction.user.mention})"
        embed = await self._update_and_get_embed(interaction, user, value, details, donation_type)
        await self._send_log_message(str(interaction.guild.id), embed, interaction)
        await interaction.followup.send(embed=embed)

    @dono.command(name="remove", description="Manually remove a donation from a user.")
    @app_commands.autocomplete(item=item_autocomplete)
    @app_commands.describe(user="The user to remove the donation from.", quantity="The amount of coins (e.g., 10m) or number of items.", item="The name of the item (optional, for item donations).")
    @app_commands.check(is_admin_or_staff)
    async def remove_donation(self, interaction: discord.Interaction, user: discord.User, quantity: str, item: Optional[str] = None):
        await interaction.response.defer() # Public response

        value, details, donation_type, error = self._calculate_donation_value(quantity, item)
        if error:
            await interaction.followup.send(error, ephemeral=True)
            return

        details += f" (Manually Removed by {interaction.user.mention})"
        embed = await self._update_and_get_embed(interaction, user, -value, details, donation_type)
        await self._send_log_message(str(interaction.guild.id), embed, interaction)
        await interaction.followup.send(embed=embed)

    # --- Core Donation Handling Logic ---
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.id == DANK_MEMER_ID:
            await self.handle_donation_message(message)

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        if after.author.id == DANK_MEMER_ID:
            await self.handle_donation_message(after)

    async def handle_donation_message(self, message: discord.Message):
        if not message.guild: return
        guild_id = str(message.guild.id)

        guild_config = self.guild_configs.get(guild_id, {})
        if message.channel.id not in guild_config.get("tracked_channel_ids", []):
            return
        
        logger.debug(f"Processing message {message.id} in tracked channel {message.channel.id}")

        user = await self.get_donator_from_message(message)
        if not user:
            logger.warning(f"Could not determine donator for message {message.id}.")
            return

        full_text = await self.extract_text_from_message(message)
        if not full_text:
            logger.debug(f"No processable text found in message {message.id}.")
            return

        donation_value = 0
        donation_details = ""
        donation_type = "Unknown"

        item_match = ITEM_DONATION_REGEX.search(full_text)
        if item_match:
            quantity_str, item_name_str = item_match.groups()
            quantity = int(quantity_str.replace(",", ""))
            matched_item = self.find_item(item_name_str)
            if matched_item:
                item_value = matched_item.get('value', 0)
                donation_value = quantity * item_value
                donation_details = f"**{quantity}x {matched_item['name']}** (Value: {donation_value:,})"
                donation_type = "item"
                logger.info(f"Detected ITEM donation: {quantity}x {matched_item['name']} from {user.name} ({user.id})")
        
        else:
            coin_match = COIN_DONATION_REGEX.search(full_text)
            if coin_match:
                amount = parse_dank_memer_amount(coin_match.group(1))
                if amount > 0:
                    donation_value = amount
                    donation_details = f"**{amount:,}** coins"
                    donation_type = "dmc"
                    logger.info(f"Detected COIN donation: {amount} from {user.name} ({user.id})")

        if donation_value > 0:
            await self.process_donation(message, user, donation_value, donation_details, donation_type)
        else:
            logger.debug(f"Message {message.id} from Dank Memer did not match any donation patterns. Text: '{full_text}'")

    async def get_donator_from_message(self, message: discord.Message) -> Optional[discord.User]:
        if hasattr(message, 'interaction_metadata') and message.interaction_metadata is not None:
            return message.interaction_metadata.user
        
        if message.mentions:
            return message.mentions[0]
            
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            return message.reference.resolved.author
            
        logger.debug(f"Could not find donator. Mentions: {message.mentions}, Reference: {message.reference}")
        return None

    async def extract_text_from_message(self, message: discord.Message) -> str:
        texts = []
        if message.content:
            texts.append(message.content)

        try:
            route = discord.http.Route('GET', '/channels/{channel_id}/messages/{message_id}', channel_id=message.channel.id, message_id=message.id)
            raw_json = await self.bot.http.request(route)
        except discord.HTTPException as e:
            logger.error(f"Failed to fetch raw message JSON for {message.id}: {e}")
            for embed in message.embeds:
                if embed.description: texts.append(embed.description)
            return normalize_text(" ".join(texts))

        def scan_component(component: Dict[str, Any]):
            for field in ['label', 'content', 'value', 'placeholder', 'description', 'title']:
                if field in component and isinstance(component.get(field), str):
                    texts.append(component[field])
            if 'options' in component:
                for option in component['options']:
                    scan_component(option)
            if 'components' in component:
                for child in component['components']:
                    scan_component(child)
        
        if 'embeds' in raw_json:
            for embed in raw_json['embeds']:
                scan_component(embed)

        if 'components' in raw_json:
            for component in raw_json['components']:
                scan_component(component)

        return normalize_text(" ".join(texts))

    def find_item(self, search_name: str) -> Optional[Dict[str, Any]]:
        normalized_search = normalize_text(search_name)
        if not normalized_search or not self.item_data: return None
        
        best_match, score, _ = process.extractOne(normalized_search, self.item_data.keys(), scorer=fuzz.WRatio)
        
        if score >= FUZZY_MATCH_THRESHOLD:
            logger.debug(f"Fuzzy matched '{search_name}' to '{best_match}' with score {score}")
            return self.item_data[best_match]
        
        logger.warning(f"Could not find a confident match for item '{search_name}'. Best guess: {best_match} ({score}%)")
        return None

    # --- Refactored Core Logic ---
    def _calculate_donation_value(self, quantity: str, item: Optional[str]) -> Tuple[int, str, str, Optional[str]]:
        """Calculates value for manual commands. Returns (value, details, type, error_msg)."""
        if item is None:
            donation_value = parse_dank_memer_amount(quantity)
            if donation_value <= 0:
                return 0, "", "", "Invalid coin quantity. Please use a number or shorthand (e.g., 10m, 2.5b)."
            donation_details = f"**{donation_value:,}** coins"
            return donation_value, donation_details, "DMC", None
        else:
            try:
                item_quantity = int(quantity)
                if item_quantity <= 0: raise ValueError()
            except ValueError:
                return 0, "", "", "Invalid item quantity. Please provide a whole number greater than 0."

            matched_item = self.find_item(item)
            if not matched_item:
                return 0, "", "", f"Could not find an item matching '{item}'. Please check the name and try again."
            
            item_value = matched_item.get('value', 0)
            donation_value = item_quantity * item_value
            donation_details = f"**{item_quantity}x {matched_item['name']}**"
            return donation_value, donation_details, "Item", None

    async def _update_and_get_embed(self, context: Union[discord.Interaction, discord.Message], user: discord.User, value: int, details: str, donation_type: str) -> discord.Embed:
        """Core logic to update donation data and create the confirmation embed."""
        user_id = str(user.id)
        guild_id = str(context.guild.id)
        
        all_donations = await load_json_async(USER_DONATIONS_FILE, {})
        user_data = all_donations.get(user_id, {"total_donation_value": 0})
        user_data["total_donation_value"] += value
        all_donations[user_id] = user_data
        await save_json_async(USER_DONATIONS_FILE, all_donations)
        self.user_donations_cache = all_donations
        logger.info(f"Instantly processed and saved donation for user {user_id}. Change: {value}")

        total_value = user_data["total_donation_value"]
        
        guild_ranks = self.guild_configs.get(guild_id, {}).get("donation_ranks", [])
        guild_ranks.sort(key=lambda x: x['value'])

        current_rank, next_rank = None, None
        if guild_ranks:
            for rank in reversed(guild_ranks):
                if total_value >= rank.get('value', 0):
                    current_rank = rank
                    break
            
            current_rank_index = guild_ranks.index(current_rank) if current_rank in guild_ranks else -1
            next_rank_index = current_rank_index + 1
            
            if next_rank_index < len(guild_ranks):
                next_rank = guild_ranks[next_rank_index]
            else:
                next_rank = None

        action = "Added" if value >= 0 else "Removed"
        
        embed = discord.Embed(
            title=f"{user.display_name}'s Donation Info - {donation_type.upper()}",
            color=discord.Color.from_rgb(47, 224, 138) if value >= 0 else discord.Color.red()
        )
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.add_field(name=f"Donation {action}", value=details, inline=False)
        embed.add_field(name="New Donation Value", value=f"⏣ {total_value:,}", inline=False)

        if next_rank and 'name' in next_rank and 'value' in next_rank:
            start_of_rank_value = current_rank['value'] if current_rank and 'value' in current_rank else 0
            progress_in_rank = total_value - start_of_rank_value
            rank_range = next_rank['value'] - start_of_rank_value
            percentage = (progress_in_rank / rank_range) * 100 if rank_range > 0 else 100.0
            bar = "🟩" * int(percentage / 10) + "⬛" * (10 - int(percentage / 10))
            embed.add_field(
                name=f"Next Rank ▶ {next_rank['name']}",
                value=f"`{bar}` **{percentage:.1f}%**\nGoal: {next_rank['value']:,}",
                inline=False
            )

        embed.set_footer(text=f"Total Donations: ⏣ {total_value:,}")
        return embed
    
    async def _send_log_message(self, guild_id: str, embed: discord.Embed, context: Union[discord.Interaction, discord.Message]):
        """Sends a message to the configured log channel, if it exists."""
        guild_config = self.guild_configs.get(guild_id, {})
        log_channel_id = guild_config.get("log_channel_id")
        if log_channel_id:
            log_embed = embed.copy()
            log_embed.set_footer(text=f"in #{context.channel.name}")
            log_embed.timestamp = discord.utils.utcnow()

            try:
                log_channel = await self.bot.fetch_channel(log_channel_id)
                await log_channel.send(embed=log_embed)
            except (discord.NotFound, discord.Forbidden) as e:
                logger.error(f"Could not send to log channel {log_channel_id}: {e}")

    async def process_donation(self, message: discord.Message, user: discord.User, value: int, details: str, donation_type: str):
        """Processes an automatic donation detection and sends a public confirmation."""
        embed = await self._update_and_get_embed(message, user, value, details, donation_type)
        try:
            await message.channel.send(content=user.mention, embed=embed)
            await message.add_reaction("✅")
            await self._send_log_message(str(message.guild.id), embed, message)
        except discord.HTTPException as e:
            logger.error(f"Failed to send confirmation message or react in #{message.channel.name}: {e}")
    
    # --- User-facing Commands ---
    @dono.command(name="view", description="Check a user's donation statistics.")
    @app_commands.describe(user="The user to check. Defaults to yourself.")
    async def donations(self, interaction: discord.Interaction, user: Optional[discord.User] = None):
        target_user = user or interaction.user
        user_id = str(target_user.id)
        user_data = self.user_donations_cache.get(user_id)
        
        if not user_data:
            embed = discord.Embed(description=f"{target_user.mention} has not made any donations yet.", color=discord.Color.orange())
            await interaction.response.send_message(embed=embed)
            return
            
        total_value = user_data.get("total_donation_value", 0)
        embed = discord.Embed(title=f"{target_user.display_name}'s Donation Stats", color=discord.Color.blue())
        embed.set_thumbnail(url=target_user.display_avatar.url)
        embed.add_field(name="Total Donation Value", value=f"⏣ {total_value:,}", inline=False)
        await interaction.response.send_message(embed=embed)
        
    @dono.command(name="leaderboard", description="Shows the top donators.")
    async def leaderboard(self, interaction: discord.Interaction):
        if not self.user_donations_cache:
            await interaction.response.send_message("No donations have been recorded yet.", ephemeral=True)
            return

        sorted_donators = sorted(self.user_donations_cache.items(), key=lambda item: item[1].get('total_donation_value', 0), reverse=True)
        
        view = LeaderboardView(self.bot, sorted_donators)
        embed = await view.get_page_embed()
        await interaction.response.send_message(embed=embed, view=view)

async def setup(bot: commands.Bot):
    """The setup function to add the cog to the bot."""
    await bot.add_cog(DonationManagerCog(bot))

# ---------------------------------
# Standalone Execution
# ---------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    if not BOT_TOKEN:
        raise ValueError("DISCORD_BOT_TOKEN must be in your .env file.")

    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    intents.guilds = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    bot.has_been_set_up = False

    @bot.event
    async def on_ready():
        logger.info(f'Logged in as {bot.user}')
        
        if not bot.has_been_set_up:
            logger.info("Performing first-time setup...")
            await setup(bot) 
            
            await bot.tree.sync()
            
            bot.has_been_set_up = True
            logger.info("Bot is ready, cog loaded, and commands synced globally.")
        else:
            logger.info("Bot has reconnected.")

    bot.run(BOT_TOKEN)
