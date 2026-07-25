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
local advance_pending = false
local last_advance_state = nil
local advance_stage = nil

local function to_val(obj)
    if type(obj) == "table" and obj.is then return tostring(obj:to_number()) end
    return tostring(obj)
end

local function json_escape(s)
    if s == nil then return "" end
    s = tostring(s)
    return (s:gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n"):gsub("\r", "\\r"))
end

local function json_bool(value)
    return value and "true" or "false"
end

local function primitive_object_json(value, depth, seen)
    if type(value) ~= "table" or depth <= 0 then return "{}" end
    seen = seen or {}
    if seen[value] then return "{}" end
    seen[value] = true
    local parts = {}
    for key, item in pairs(value) do
        if type(key) == "string" then
            local encoded = nil
            if type(item) == "number" then
                encoded = tostring(item)
            elseif type(item) == "boolean" then
                encoded = json_bool(item)
            elseif type(item) == "string" then
                encoded = '"' .. json_escape(item) .. '"'
            elseif type(item) == "table" and depth > 1 then
                encoded = primitive_object_json(item, depth - 1, seen)
            end
            if encoded then
                table.insert(parts, '"' .. json_escape(key) .. '":' .. encoded)
            end
        end
    end
    seen[value] = nil
    return "{" .. table.concat(parts, ",") .. "}"
end

local function get_edition_name(card)
    if not card or type(card.edition) ~= "table" then return "" end
    for _, name in ipairs({"negative", "polychrome", "holo", "foil"}) do
        if card.edition[name] then return name end
    end
    return ""
end

local function hand_card_json(card, index)
    local base = card and card.base or {}
    local ability = card and card.ability or {}
    local center = card and card.config and card.config.center or {}
    return string.format(
        '{"index":%d,"value":"%s","suit":"%s","center":"%s","seal":"%s","edition":"%s","debuff":%s,"facing":"%s","forced":%s,"played_this_ante":%s,"extra_chips":%d}',
        index,
        json_escape(tostring(base.value or "")),
        json_escape(tostring(base.suit or "")),
        json_escape(tostring(center.key or "")),
        json_escape(tostring(card and card.seal or "")),
        json_escape(get_edition_name(card)),
        json_bool(card and card.debuff),
        json_escape(tostring(card and card.facing or "")),
        json_bool(ability.forced_selection),
        json_bool(ability.played_this_ante),
        tonumber(ability.perma_bonus) or 0
    )
end

local function inventory_card_json(card, index)
    local ability = card and card.ability or {}
    local center = card and card.config and card.config.center or {}
    local sell_cost = tonumber(card and card.sell_cost) or 0
    return string.format(
        '{"index":%d,"key":"%s","name":"%s","set":"%s","edition":"%s","debuff":%s,"sell_cost":%d,"ability":%s}',
        index,
        json_escape(tostring(center.key or "")),
        json_escape(tostring(ability.name or center.name or "")),
        json_escape(tostring(ability.set or center.set or "")),
        json_escape(get_edition_name(card)),
        json_bool(card and card.debuff),
        sell_cost,
        primitive_object_json(ability, 2)
    )
end

local function inventory_json(area)
    local parts = {}
    if area and type(area.cards) == "table" then
        for index, card in ipairs(area.cards) do
            table.insert(parts, inventory_card_json(card, index))
        end
    end
    return "[" .. table.concat(parts, ",") .. "]"
end

local function context_json()
    local game = G and G.GAME or {}
    local current_round = game.current_round or {}
    local round_resets = game.round_resets or {}
    local blind = game.blind or {}
    local blind_config = blind.config and blind.config.blind or {}
    local seed = game.pseudorandom and game.pseudorandom.seed or ""
    local deck_remaining = G and G.deck and G.deck.cards and #G.deck.cards or 0
    local deck_total = G and G.playing_cards and #G.playing_cards or 0
    local blind_type = ""
    if type(blind.get_type) == "function" then
        local ok, result = pcall(blind.get_type, blind)
        if ok then blind_type = tostring(result or "") end
    end
    local run_json = string.format(
        '{"seed":"%s","ante":%d,"round":%d,"stake":%d,"deck_remaining":%d,"deck_total":%d,"hands_played":%d,"discards_used":%d,"most_played_hand":"%s"}',
        json_escape(tostring(seed)),
        tonumber(round_resets.ante) or 0,
        tonumber(game.round) or 0,
        tonumber(game.stake) or 0,
        deck_remaining,
        deck_total,
        tonumber(current_round.hands_played) or 0,
        tonumber(current_round.discards_used) or 0,
        json_escape(tostring(current_round.most_played_poker_hand or ""))
    )
    local blind_json = string.format(
        '{"key":"%s","name":"%s","type":"%s","boss":%s,"disabled":%s,"debuff":%s}',
        json_escape(tostring(blind_config.key or "")),
        json_escape(tostring(blind.name or blind_config.name or "")),
        json_escape(blind_type),
        json_bool(blind.boss),
        json_bool(blind.disabled),
        primitive_object_json(blind.debuff or blind_config.debuff or {}, 2)
    )
    return string.format(
        '"state_version":2,"run":%s,"blind":%s,"jokers":%s,"consumables":%s',
        run_json,
        blind_json,
        inventory_json(G and G.jokers),
        inventory_json(G and G.consumeables)
    )
end

local function get_round_result(chips, blind_chips)
    if blind_chips > 0 and chips >= blind_chips then return "won" end
    if G and G.STATES and G.STATES.ROUND_EVAL and G.STATE == G.STATES.ROUND_EVAL then
        return "won"
    end
    if G and G.STATES and G.STATES.GAME_OVER and G.STATE == G.STATES.GAME_OVER then
        return "lost"
    end
    return ""
end

-- Transition states keep score/resource fields so Python can detect a finished blind.
local function write_state_json_minimal()
    if not G or not G.STATE then return end
    local phase_raw = to_val(G.STATE)
    local phase = tonumber(phase_raw)
    if phase == nil then phase = 0 end
    local money = 0
    local chips = 0
    local blind_chips = 0
    local hands_left = 0
    local discards_left = 0
    local last_hand_played = ""
    if G.GAME then
        money = tonumber(to_val(G.GAME.dollars)) or 0
        chips = tonumber(to_val(G.GAME.chips)) or 0
        if G.GAME.blind and G.GAME.blind.chips then
            blind_chips = tonumber(to_val(G.GAME.blind.chips)) or 0
        end
        if G.GAME.current_round then
            hands_left = tonumber(to_val(G.GAME.current_round.hands_left)) or 0
            discards_left = tonumber(to_val(G.GAME.current_round.discards_left)) or 0
        end
        if type(G.GAME.last_hand_played) == "string" then
            last_hand_played = json_escape(G.GAME.last_hand_played)
        end
    end
    local round_result = get_round_result(chips, blind_chips)
    state_seq = state_seq + 1
    local body = string.format(
        '{"seq":%d,"phase":%d,"money":%d,"chips":%d,"blind_chips":%d,"hands_left":%d,"discards_left":%d,"hand":[],"hand_levels":{},"last_hand_played":"%s","round_result":"%s",%s}',
        state_seq, phase, money, chips, blind_chips, hands_left, discards_left,
        last_hand_played, round_result, context_json()
    )
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
            table.insert(hand_parts, hand_card_json(card, i))
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
                    local played = tonumber(data.played) or 0
                    local played_this_round = tonumber(data.played_this_round) or 0
                    table.insert(out, string.format('"%s":{"level":%d,"chips":%d,"mult":%d,"played":%d,"played_this_round":%d}',
                        json_escape(tostring(name)), lvl, c, m, played, played_this_round))
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
    local round_result = get_round_result(chips, blind_chips)
    state_seq = state_seq + 1
    local body = string.format(
        '{"seq":%d,"phase":%d,"money":%d,"chips":%d,"blind_chips":%d,"hands_left":%d,"discards_left":%d,"hand":%s,"hand_levels":%s,"last_hand_played":"%s","round_result":"%s",%s}',
        state_seq, phase, money, chips, blind_chips, hands_left, discards_left,
        hand_json, hand_levels_json, last_hand_played, round_result, context_json()
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
    local current_round = G.GAME and G.GAME.current_round
    local hands_left = current_round and tonumber(to_val(current_round.hands_left)) or 0
    local discards_left = current_round and tonumber(to_val(current_round.discards_left)) or 0
    if cmd.action == "play" and hands_left <= 0 then
        print("AI: Rejected play with no hands remaining")
        return
    elseif cmd.action == "discard" and discards_left <= 0 then
        print("AI: Rejected discard with no discards remaining")
        return
    elseif cmd.action ~= "play" and cmd.action ~= "discard" then
        print("AI: Rejected unknown action")
        return
    end

    local card_limit = cmd.action == "play" and 5 or #G.hand.cards
    local legal_cards = {}
    for _, index in ipairs(cmd.cards or {}) do
        if #legal_cards >= card_limit then break end
        if type(index) == "number" and index >= 1 and index <= #G.hand.cards then
            table.insert(legal_cards, index)
        end
    end
    if #legal_cards == 0 then
        print("AI: Rejected action with no valid cards")
        return
    end
    select_cards(legal_cards)
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
    local f = io.open(command_file, "r")
    if not f then return end
    local content = f:read("*all")
    f:close()

    local action = content:match('["\']action["\']%s*:%s*["\']([^"\']+)["\']')
    if action == "advance" then
        os.remove(command_file)
        advance_pending = true
        last_advance_state = nil
        advance_stage = nil
        print("AI: Advancing to next playable hand")
        return
    end
    -- Keep play/discard commands queued until Balatro returns to hand selection.
    if not safe_to_execute_command() then return end
    os.remove(command_file)
    local cards_str = content:match('["\']cards["\']%s*:%s*%[([^%]]+)%]')
    local card_indices = {}
    if cards_str then
        for index in cards_str:gmatch("%d+") do table.insert(card_indices, tonumber(index)) end
    end
    if action then execute_command({action = action, cards = card_indices}) end
end

local function find_ui_element(id)
    if not G or not G.I or type(G.I.UIBOX) ~= "table" then return nil end
    for _, box in pairs(G.I.UIBOX) do
        if box and type(box.get_UIE_by_ID) == "function" then
            local ok, element = pcall(box.get_UIE_by_ID, box, id)
            if ok and element then return element end
        end
    end
    return nil
end

local function process_advance()
    if not advance_pending or not G or not G.STATES then return end
    if safe_to_execute_command() then
        local game = G.GAME or {}
        local chips = tonumber(to_val(game.chips)) or 0
        local blind_chips = game.blind and tonumber(to_val(game.blind.chips)) or 0
        -- A won blind can briefly return to SELECTING_HAND before ROUND_EVAL.
        -- Keep the request alive until the Cash Out screen appears.
        if blind_chips > 0 and chips >= blind_chips then return end
        advance_pending = false
        last_advance_state = nil
        advance_stage = nil
        return
    end

    if G.STATE == G.STATES.GAME_OVER then
        if advance_stage == nil
            and G.STATE_COMPLETE
            and G.OVERLAY_MENU
            and G.FUNCS.notify_then_setup_run then
            local ok, err = pcall(
                G.FUNCS.notify_then_setup_run,
                {config = {id = "from_game_over"}}
            )
            if ok then
                advance_stage = "wait_run_setup"
                print("AI: Opened New Run setup")
            elseif err then
                print("Bridge New Run error: " .. tostring(err))
            end
        elseif advance_stage == "wait_run_setup"
            and G.OVERLAY_MENU
            and G.SETTINGS.current_setup == "New Run"
            and G.GAME
            and G.GAME.viewed_back
            and G.FUNCS.start_setup_run then
            if G.P_CENTERS and G.P_CENTERS.b_red and G.GAME.viewed_back.change_to then
                G.GAME.viewed_back:change_to(G.P_CENTERS.b_red)
            end
            G.run_setup_seed = nil
            G.setup_seed = ""
            local ok, err = pcall(
                G.FUNCS.start_setup_run,
                {config = {id = "bridge_auto_play"}}
            )
            if ok then
                advance_stage = "run_starting"
                print("AI: Selected Red Deck and pressed Play")
            elseif err then
                print("Bridge Play error: " .. tostring(err))
            end
        end
        return
    end

    -- GAME_OVER uses advance_stage while the run setup overlay changes. Other
    -- transitions are guarded by last_advance_state until G.STATE changes.
    advance_stage = nil
    if last_advance_state == G.STATE then return end

    local cash_out_button = G.STATE == G.STATES.ROUND_EVAL
        and find_ui_element("cash_out_button") or nil
    if cash_out_button
        and cash_out_button.config
        and cash_out_button.config.button == "cash_out"
        and G.FUNCS.cash_out then
        -- Calling UIElement:click() can silently do nothing while the button is
        -- still becoming visible. Its presence means round evaluation has
        -- finished building, so invoke the same callback with the real element.
        local ok, err = pcall(G.FUNCS.cash_out, cash_out_button)
        if ok then
            last_advance_state = G.STATE
            print("AI: Pressed Cash Out")
        elseif err then
            print("Bridge Cash Out error: " .. tostring(err))
        end
    elseif G.STATE == G.STATES.SHOP and G.shop and G.FUNCS.toggle_shop then
        local ok, err = pcall(G.FUNCS.toggle_shop, {config = {}})
        if ok then
            last_advance_state = G.STATE
        elseif err then
            print("Bridge Next Round error: " .. tostring(err))
        end
    elseif G.STATE == G.STATES.BLIND_SELECT
        and G.blind_select
        and G.blind_select_opts
        and G.FUNCS.select_blind then
        local blind_type = G.GAME and G.GAME.blind_on_deck or "Small"
        local blind_pane = G.blind_select_opts[string.lower(blind_type)]
        local select_button = blind_pane
            and blind_pane.get_UIE_by_ID
            and blind_pane:get_UIE_by_ID("select_blind_button") or nil
        if select_button then
            local ok, err = pcall(G.FUNCS.select_blind, select_button)
            if ok then
                last_advance_state = G.STATE
                print("AI: Selected " .. tostring(blind_type) .. " Blind")
            elseif err then
                print("Bridge Select Blind error: " .. tostring(err))
            end
        end
    end
end

-- MAIN LOOP
local game_update_ref = Game.update
function Game:update(dt)
    game_update_ref(self, dt)

    -- 1. Check for incoming AI commands
    check_for_commands()
    process_advance()

    -- 2. Event-based logic: Only snapshot when we are in a "playable" state
    -- G.STATES.SELECTING_HAND (1) is when cards are in hand and UI is ready
    if G.STATE == G.STATES.SELECTING_HAND then
        
        local current_hands = G.GAME.current_round and G.GAME.current_round.hands_left or 0
        local current_discards = G.GAME.current_round and G.GAME.current_round.discards_left or 0
        local current_card_count = (G.hand and G.hand.cards) and #G.hand.cards or 0

        -- TRIGGER CONDITIONS:
        -- A) We just entered the round (State changed from something else to 1)
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
