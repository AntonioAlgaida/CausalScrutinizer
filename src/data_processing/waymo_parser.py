import numpy as np
from typing import Dict, List

# --- NEW: Define the map feature enums for clarity ---
# Based on the official Waymo Open Motion Dataset documentation
MAP_FEATURE_TYPES = {
    # Lanes
    'TYPE_LANE_FREEWAY': 1,
    'TYPE_LANE_SURFACE_STREET': 2,
    'TYPE_LANE_BIKE_LANE': 3,
    
    # Lines
    'TYPE_ROAD_LINE_BROKEN_SINGLE_WHITE': 11,
    'TYPE_ROAD_LINE_SOLID_SINGLE_WHITE': 12,
    'TYPE_ROAD_LINE_SOLID_DOUBLE_WHITE': 13,
    'TYPE_ROAD_LINE_BROKEN_SINGLE_YELLOW': 14,
    'TYPE_ROAD_LINE_BROKEN_DOUBLE_YELLOW': 15,
    'TYPE_ROAD_LINE_SOLID_SINGLE_YELLOW': 16,
    'TYPE_ROAD_LINE_SOLID_DOUBLE_YELLOW': 17,
    'TYPE_ROAD_LINE_PASSING_DOUBLE_YELLOW': 18,
    
    # Edges
    'TYPE_ROAD_EDGE_BOUNDARY': 21,
    'TYPE_ROAD_EDGE_MEDIAN': 22,
    
    # Signs & Structures
    'TYPE_STOP_SIGN': 31,
    'TYPE_CROSSWALK': 41,
    'TYPE_SPEED_BUMP': 51,
    'TYPE_DRIVEWAY': 61
}

def _parse_map_layers_v4(polylines: np.ndarray, polyline_types: np.ndarray) -> Dict[str, List[np.ndarray]]:
    """
    Processes raw map polylines and sorts them into a deeply semantic
    dictionary of map layers, with individual line types separated.
    """
    # Initialize the new, more detailed data structure
    map_layers = {
        'surface_street_lanes': [], 'freeway_lanes': [], 'bike_lanes': [], 'driveways': [],
        'crosswalks': [], 'speed_bumps': [],
        'lines_white_dashed': [], 'lines_white_solid_single': [], 'lines_white_solid_double': [],
        'lines_yellow_dashed': [], 'lines_yellow_solid_single': [], 'lines_yellow_solid_double': [],
        'lines_yellow_passing_double': [], 'lines_yellow_broken_double': [],
        'road_edges': [], 'medians': [],
        'stop_signs': []
    }

    for i, p_type in enumerate(polyline_types):
        polyline = polylines[i]
        
        # Surfaces
        if p_type == MAP_FEATURE_TYPES['TYPE_LANE_SURFACE_STREET']:
            map_layers['surface_street_lanes'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_LANE_FREEWAY']:
            map_layers['freeway_lanes'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_LANE_BIKE_LANE']:
            map_layers['bike_lanes'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_DRIVEWAY']:
            map_layers['driveways'].append(polyline)
            
        # Markings
        elif p_type == MAP_FEATURE_TYPES['TYPE_CROSSWALK']:
            map_layers['crosswalks'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_SPEED_BUMP']:
            map_layers['speed_bumps'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_BROKEN_SINGLE_WHITE']:
            map_layers['lines_white_dashed'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_SOLID_SINGLE_WHITE']:
            map_layers['lines_white_solid_single'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_SOLID_DOUBLE_WHITE']:
            map_layers['lines_white_solid_double'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_BROKEN_SINGLE_YELLOW']:
            map_layers['lines_yellow_dashed'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_BROKEN_DOUBLE_YELLOW']:
            map_layers['lines_yellow_broken_double'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_SOLID_SINGLE_YELLOW']:
            map_layers['lines_yellow_solid_single'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_SOLID_DOUBLE_YELLOW']:
            map_layers['lines_yellow_solid_double'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_LINE_PASSING_DOUBLE_YELLOW']:
            map_layers['lines_yellow_passing_double'].append(polyline)
            
        # Boundaries
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_EDGE_BOUNDARY']:
            map_layers['road_edges'].append(polyline)
        elif p_type == MAP_FEATURE_TYPES['TYPE_ROAD_EDGE_MEDIAN']:
            map_layers['medians'].append(polyline)
            
        # Objects
        elif p_type == MAP_FEATURE_TYPES['TYPE_STOP_SIGN']:
            map_layers['stop_signs'].append(polyline[0, :3])

    return map_layers

# --- NEW: Function to create the Lane ID to Polyline mapping ---
def _create_lane_id_mapping(polylines: np.ndarray, polyline_types: np.ndarray, polyline_ids: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Creates a dictionary that maps a lane's unique ID to its polyline data.
    """
    lane_id_to_polyline = {}
    lane_types = {
        MAP_FEATURE_TYPES['TYPE_LANE_FREEWAY'],
        MAP_FEATURE_TYPES['TYPE_LANE_SURFACE_STREET'],
        MAP_FEATURE_TYPES['TYPE_LANE_BIKE_LANE']
    }

    for i, id in enumerate(polyline_ids):
        # We only care about polylines that are actual lanes
        if polyline_types[i] in lane_types:
            lane_id_to_polyline[id] = polylines[i]
            
    return lane_id_to_polyline

def load_npz_scenario_v5(file_path: str) -> Dict[str, any]:
    """
    --- V5: Final Parser with Lane ID Mapping ---
    Loads a scenario and enriches it with a structured map AND a lane_id lookup table.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Error loading .npz file {file_path}: {e}")
        raise

    # --- Load all base keys ---
    base_keys = [
        'scenario_id', 'all_agent_trajectories', 'sdc_track_index',
        'object_types', 'valid_mask', 'dynamic_map_states', 'sdc_route',
        'map_polylines', 'map_polyline_types', 'map_polyline_ids', 'object_ids'
    ]
    scenario_data = {key: data[key].item() if data[key].ndim == 0 else data[key] for key in base_keys}

    # --- Run our two parsing functions ---
    
    # 1. Parse into semantic layers (for general drawing)
    scenario_data['map_layers'] = _parse_map_layers_v4(
        scenario_data['map_polylines'],
        scenario_data['map_polyline_types']
    )

    # 2. NEW: Create the lane_id -> polyline lookup table (for traffic lights)
    scenario_data['lane_id_to_polyline'] = _create_lane_id_mapping(
        scenario_data['map_polylines'],
        scenario_data['map_polyline_types'],
        scenario_data['map_polyline_ids']
    )
    
    # We can now delete the raw map data to keep the final dictionary clean
    del scenario_data['map_polylines']
    del scenario_data['map_polyline_types']
    del scenario_data['map_polyline_ids']

    return scenario_data

# --- Main public function for backward compatibility ---
def load_npz_scenario(file_path: str) -> Dict[str, any]:
    """
    Loads a pre-processed Waymo scenario from a .npz file.
    This is the main entry point, now calling the V5 parser.
    """
    return load_npz_scenario_v5(file_path)