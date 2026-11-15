"""Every action the viewer understands. Key bindings live in configs/viz/default.yaml."""

COMMANDS: tuple[str, ...] = (
    # general
    "quit", "help",
    # playback
    "play_pause", "step_forward", "step_back", "jump_start", "jump_end",
    "speed_up", "speed_down", "next_event", "prev_event",
    # overlays
    "toggle_fog", "toggle_belief", "toggle_vision", "toggle_trails", "toggle_probabilities",
    "toggle_arrows", "toggle_labels", "toggle_coordinates", "toggle_spawns",
    "cycle_view_team", "next_agent", "prev_agent", "deselect",
    # live play
    "new_episode", "new_seed", "swap_sides", "cycle_red_controller", "cycle_blue_controller",
    "toggle_autoloop",
    # map maker
    "toggle_edit", "tool_wall", "tool_erase", "tool_red_base", "tool_blue_base",
    "tool_red_spawn", "tool_blue_spawn", "toggle_mirror", "grow_width", "shrink_width",
    "grow_height", "shrink_height", "clear_map", "validate_map", "save_map", "next_map",
    "prev_map", "play_on_map",
    # export
    "screenshot", "export_gif", "export_mp4", "save_recording",
)

# Short human-readable descriptions for the help screen and button tooltips.
DESCRIPTIONS: dict[str, str] = {
    "quit": "Quit", "help": "Show / hide this help",
    "play_pause": "Play / pause", "step_forward": "Step forward", "step_back": "Step back",
    "jump_start": "Jump to start", "jump_end": "Jump to end", "speed_up": "Faster", "speed_down": "Slower",
    "next_event": "Next event (pickup / tag / capture)", "prev_event": "Previous event",
    "toggle_fog": "Fog of war (what the viewed team knows)", "toggle_belief": "Belief map (where enemies were last seen)",
    "toggle_vision": "Vision range", "toggle_trails": "Movement trails", "toggle_probabilities": "Action probabilities (selected agent)",
    "toggle_arrows": "Chosen-action arrows", "toggle_labels": "Agent labels (id + role)", "toggle_coordinates": "Cell coordinates",
    "toggle_spawns": "Spawn markers", "cycle_view_team": "View as RED / BLUE / everyone", "next_agent": "Select next agent",
    "prev_agent": "Select previous agent", "deselect": "Clear selection",
    "new_episode": "New episode (same seed)", "new_seed": "New episode (new seed)", "swap_sides": "Swap the two controllers",
    "cycle_red_controller": "Next RED controller", "cycle_blue_controller": "Next BLUE controller", "toggle_autoloop": "Auto-restart finished episodes",
    "toggle_edit": "Map maker on / off", "tool_wall": "Tool: wall", "tool_erase": "Tool: erase", "tool_red_base": "Tool: RED base",
    "tool_blue_base": "Tool: BLUE base", "tool_red_spawn": "Tool: RED spawn", "tool_blue_spawn": "Tool: BLUE spawn",
    "toggle_mirror": "Mirror edits left-right", "grow_width": "Wider map", "shrink_width": "Narrower map",
    "grow_height": "Taller map", "shrink_height": "Shorter map", "clear_map": "Clear all walls", "validate_map": "Check the map",
    "save_map": "Save map to file", "next_map": "Load next map file", "prev_map": "Load previous map file", "play_on_map": "Play on this map",
    "screenshot": "Save a screenshot", "export_gif": "Export episode as GIF", "export_mp4": "Export episode as MP4",
    "save_recording": "Save episode recording (JSON)",
}
assert set(DESCRIPTIONS) == set(COMMANDS)

# How the help screen groups the commands (explicit, so adding a command cannot silently mis-group).
GROUPS: dict[str, tuple[str, ...]] = {
    "Playback": ("play_pause", "step_forward", "step_back", "jump_start", "jump_end", "speed_up", "speed_down",
                 "next_event", "prev_event"),
    "Overlays": ("toggle_fog", "toggle_belief", "toggle_vision", "toggle_trails", "toggle_probabilities",
                 "toggle_arrows", "toggle_labels", "toggle_coordinates", "toggle_spawns", "cycle_view_team",
                 "next_agent", "prev_agent", "deselect"),
    "Live play": ("new_episode", "new_seed", "swap_sides", "cycle_red_controller", "cycle_blue_controller",
                  "toggle_autoloop"),
    "Map maker": ("toggle_edit", "tool_wall", "tool_erase", "tool_red_base", "tool_blue_base", "tool_red_spawn",
                  "tool_blue_spawn", "toggle_mirror", "grow_width", "shrink_width", "grow_height", "shrink_height",
                  "clear_map", "validate_map", "save_map", "next_map", "prev_map", "play_on_map"),
    "Export": ("screenshot", "export_gif", "export_mp4", "save_recording"),
    "General": ("help", "quit"),
}
assert sorted(c for g in GROUPS.values() for c in g) == sorted(COMMANDS)
