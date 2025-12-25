-- F:\Github\Balatro-RL\bridge\main.lua
local command_file = "F:\\Github\\Balatro-RL\\bridge\\command.json"

-- Track values to detect changes
local last_state = nil
local last_hands = nil
local last_discards = nil
local last_card_count = 0

local function to_val(obj)
    if type(obj) == "table" and obj.is then return tostring(obj:to_number()) end
    return tostring(obj)
end

local function select_cards(indices)
    if not G.hand or not G.hand.cards then return end
    G.hand:unhighlight_all()
    for _, index in ipairs(indices) do
        local card = G.hand.cards[index]
        if card then G.hand:add_to_highlighted(card) end
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

local function execute_command(cmd)
    if cmd.cards then select_cards(cmd.cards) end

    if cmd.action == "play" and G.FUNCS.play_cards_from_highlighted then
        print("AI: Executing Play")
        G.FUNCS.play_cards_from_highlighted()
    elseif cmd.action == "discard" and G.FUNCS.discard_cards_from_highlighted then
        print("AI: Executing Discard")
        G.FUNCS.discard_cards_from_highlighted()
    end
end

local function check_for_commands()
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

            -- Update trackers
            last_state = G.STATE
            last_hands = current_hands
            last_discards = current_discards
            last_card_count = current_card_count
        end
    else
        -- Reset trackers when leaving the round so we trigger again next time we enter
        if last_state ~= G.STATE then
            last_state = G.STATE
        end
    end
end