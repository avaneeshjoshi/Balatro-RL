-- F:\Github\Balatro-RL\bridge\main.lua
-- IMPORTANT: This file must write "seq" (incrementing) in every state.json so the Python env
-- can tell new state from stale. Restart Balatro after editing so the game loads this version.
local bridge_dir = "F:\\Github\\Balatro-RL\\bridge"
local command_file = bridge_dir .. "\\command.json"
local state_file = bridge_dir .. "\\state.json"

-- Track values to detect changes
local last_state = nil
local last_hands = nil
local last_discards = nil
local last_card_count = 0
local state_seq = 0  -- Increments each state write; agent waits for seq > last_seq

local function to_val(obj)
    if type(obj) == "table" and obj.is then return tostring(obj:to_number()) end
    return tostring(obj)
end

local function json_escape(s)
    if s == nil then return "" end
    s = tostring(s)
    return (s:gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n"):gsub("\r", "\\r"))
end

-- Minimal state write (only needs G and G.STATE). Use when leaving SELECTING_HAND so the agent always gets a new seq.
local function write_state_json_minimal()
    if not G or not G.STATE then return end
    local phase_raw = to_val(G.STATE)
    local phase = tonumber(phase_raw)
    if phase == nil then phase = 0 end
    state_seq = state_seq + 1
    local body = string.format('{"seq":%d,"phase":%d,"money":0,"chips":0,"blind_chips":0,"hands_left":0,"discards_left":0,"hand":[],"hand_levels":{}}', state_seq, phase)
    local f = io.open(state_file, "w")
    if f then
        f:write(body)
        f:close()
    end
end

local function write_state_json()
    if not G or not G.GAME or not G.STATE then return end
    local phase_raw = to_val(G.STATE)
    local phase = tonumber(phase_raw)
    if phase == nil then phase = 0 end
    local money = tonumber(to_val(G.GAME.dollars)) or 0
    local chips = tonumber(to_val(G.GAME.chips)) or 0
    local blind_chips = 0
    if G.GAME.blind and G.GAME.blind.chips then
        blind_chips = tonumber(to_val(G.GAME.blind.chips)) or 0
    end
    local hands_left = 0
    local discards_left = 0
    if G.GAME.current_round then
        hands_left = tonumber(to_val(G.GAME.current_round.hands_left)) or 0
        discards_left = tonumber(to_val(G.GAME.current_round.discards_left)) or 0
    end
    local hand_parts = {}
    if G.hand and G.hand.cards then
        for i, card in ipairs(G.hand.cards) do
            local val = tostring(card.base.value or "")
            local suit = tostring(card.base.suit or "")
            table.insert(hand_parts, string.format('{"index":%d,"value":"%s","suit":"%s"}', i, json_escape(val), json_escape(suit)))
        end
    end
    local hand_json = "[" .. table.concat(hand_parts, ",") .. "]"
    -- Hand levels: G.GAME.hands[key] often has .level, .chips, .mult (Balatro hand table)
    local hand_levels_json = "{}"
    if G.GAME and G.GAME.hands and type(G.GAME.hands) == "table" then
        local ok, parts = pcall(function()
            local out = {}
            for name, data in pairs(G.GAME.hands) do
                if type(data) == "table" then
                    local lvl = (data.level or data.lvl or data.level_override or 0)
                    local c = (data.chips or data.chip or 0)
                    local m = (data.mult or data.mult_mod or 0)
                    if type(lvl) ~= "number" then lvl = tonumber(tostring(lvl)) or 0 end
                    if type(c) ~= "number" then c = tonumber(tostring(c)) or 0 end
                    if type(m) ~= "number" then m = tonumber(tostring(m)) or 0 end
                    table.insert(out, string.format('"%s":{"level":%d,"chips":%d,"mult":%d}',
                        json_escape(tostring(name)), lvl, c, m))
                end
            end
            return out
        end)
        if ok and parts and #parts > 0 then
            hand_levels_json = "{" .. table.concat(parts, ",") .. "}"
        end
    end
    -- Last played hand type (set in state_events.lua when you play); used for reward shaping
    local last_hand_played = ""
    if G.GAME and G.GAME.last_hand_played and type(G.GAME.last_hand_played) == "string" then
        last_hand_played = json_escape(G.GAME.last_hand_played)
    end
    state_seq = state_seq + 1
    local body = string.format(
        '{"seq":%d,"phase":%d,"money":%d,"chips":%d,"blind_chips":%d,"hands_left":%d,"discards_left":%d,"hand":%s,"hand_levels":%s,"last_hand_played":"%s"}',
        state_seq, phase, money, chips, blind_chips, hands_left, discards_left, hand_json, hand_levels_json, last_hand_played
    )
    local f = io.open(state_file, "w")
    if f then
        f:write(body)
        f:close()
    end
end

local function select_cards(indices)
    if not G or not G.hand or not G.hand.cards or type(G.hand.cards) ~= "table" then return end
    local n = #G.hand.cards
    if n == 0 then return end
    G.hand:unhighlight_all()
    for _, index in ipairs(indices) do
        if type(index) == "number" and index >= 1 and index <= n then
            local card = G.hand.cards[index]
            if card then G.hand:add_to_highlighted(card) end
        end
    end
end

local function dump_game_state()
    if not G or not G.GAME or not G.STATE then return end

    print("--- EVENT TRIGGERED SNAPSHOT ---")
    print(string.format("PHASE: %s | MONEY: $%s | CHIPS: %s/%s", 
        to_val(G.STATE), to_val(G.GAME.dollars), to_val(G.GAME.chips), 
        (G.GAME.blind and G.GAME.blind.chips) and to_val(G.GAME.chips) or "0"))
    
    local hands_left = G.GAME.current_round and G.GAME.current_round.hands_left or 0
    local discards_left = G.GAME.current_round and G.GAME.current_round.discards_left or 0
    print(string.format("HANDS LEFT: %s | DISCARDS LEFT: %s", to_val(hands_left), to_val(discards_left)))
    
    local hand_str = ""
    if G.hand and G.hand.cards then
        for i, card in ipairs(G.hand.cards) do
            hand_str = hand_str .. string.format("%d:[%s of %s] ", i, tostring(card.base.value), tostring(card.base.suit))
        end
    end
    print("HAND: " .. (hand_str ~= "" and hand_str or "Empty"))
    print("--------------------------------")
end

local function safe_to_execute_command()
    if not G or not G.STATES or G.STATE ~= G.STATES.SELECTING_HAND then return false end
    if not G.hand or not G.hand.cards or type(G.hand.cards) ~= "table" then return false end
    if #G.hand.cards == 0 then return false end
    return true
end

local function execute_command(cmd)
    if not safe_to_execute_command() then return end
    if cmd.cards then select_cards(cmd.cards) end
    if not safe_to_execute_command() then return end

    local ok, err
    if cmd.action == "play" and G.FUNCS and G.FUNCS.play_cards_from_highlighted then
        print("AI: Executing Play")
        ok, err = pcall(G.FUNCS.play_cards_from_highlighted)
    elseif cmd.action == "discard" and G.FUNCS and G.FUNCS.discard_cards_from_highlighted then
        print("AI: Executing Discard")
        ok, err = pcall(G.FUNCS.discard_cards_from_highlighted)
    end
    if ok == false and err then print("Bridge execute_command error: " .. tostring(err)) end
end

local function check_for_commands()
    -- Only process commands when game is in hand selection with valid hand (avoids state_events.lua ipairs(nil)).
    if not safe_to_execute_command() then return end
    local f = io.open(command_file, "r")
    if not f then return end
    local content = f:read("*all")
    f:close()
    os.remove(command_file)

    local action = content:match('["\']action["\']%s*:%s*["\']([^"\']+)["\']')
    local cards_str = content:match('["\']cards["\']%s*:%s*%[([^%]]+)%]')
    local card_indices = {}
    if cards_str then
        for index in cards_str:gmatch("%d+") do table.insert(card_indices, tonumber(index)) end
    end
    if action then execute_command({action = action, cards = card_indices}) end
end

-- MAIN LOOP
local game_update_ref = Game.update
function Game:update(dt)
    game_update_ref(self, dt)

    -- 1. Check for incoming AI commands
    check_for_commands()

    -- 2. Event-based logic: Only snapshot when we are in a "playable" state
    -- G.STATES.SELECTING_HAND (4) is when cards are in hand and UI is ready
    if G.STATE == G.STATES.SELECTING_HAND then
        
        local current_hands = G.GAME.current_round and G.GAME.current_round.hands_left or 0
        local current_discards = G.GAME.current_round and G.GAME.current_round.discards_left or 0
        local current_card_count = (G.hand and G.hand.cards) and #G.hand.cards or 0

        -- TRIGGER CONDITIONS:
        -- A) We just entered the round (State changed from something else to 4)
        -- B) We just played a hand (Hands left decreased)
        -- C) We just discarded (Discards left decreased)
        -- D) Hand size changed (e.g. after drawing new cards)
        if (G.STATE ~= last_state) or 
           (current_hands ~= last_hands) or 
           (current_discards ~= last_discards) or
           (current_card_count ~= last_card_count) then
            
            -- Wait a tiny bit for animations to settle if needed, but pcall is safer
            pcall(dump_game_state)
            pcall(write_state_json)

            -- Update trackers
            last_state = G.STATE
            last_hands = current_hands
            last_discards = current_discards
            last_card_count = current_card_count
        end
    else
        -- When leaving SELECTING_HAND (e.g. after play/discard), write minimal state so the agent
        -- always gets a new seq (hand=[]). Uses minimal write so we don't depend on G.GAME during transitions.
        if last_state ~= G.STATE then
            pcall(write_state_json_minimal)
            last_state = G.STATE
        end
    end
end