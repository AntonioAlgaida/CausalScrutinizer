# File: src/rendering/scenario_renderer.py

import pygame
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
# --- UPDATED: We now assume the V2 parser is being used ---
from src.data_processing.waymo_parser import load_npz_scenario


class ScenarioRenderer:
    """
    --- V2 High-Fidelity Renderer ---
    Renders a Waymo scenario from parsed NumPy data into a top-down GIF.
    Differentiates between agent types for improved VLM context.
    """

    def __init__(self, scenario_data: Dict[str, np.ndarray], config: Dict = None):
        self.scenario_data = scenario_data
        self.config = self._get_default_config()
    
        if config:
            self.config.update(config)

        self.screen_width = 896
        self.screen_height = 896

        # --- MODIFIED: Conditional Initialization Logic ---
        # Check if scenario_data is provided and has the necessary keys.
        if scenario_data and 'sdc_track_index' in scenario_data:
            # --- Full setup for rendering a real scenario ---
            sdc_index = self.scenario_data['sdc_track_index']
            sdc_traj = self.scenario_data['all_agent_trajectories'][sdc_index]
            sdc_valid_mask = self.scenario_data['valid_mask'][sdc_index]
            
            first_valid_idx = np.where(sdc_valid_mask)[0]
            if len(first_valid_idx) == 0:
                raise ValueError("SDC has no valid timesteps in this scenario.")
            
            self.scene_center_world = sdc_traj[first_valid_idx[40], :2]
        
        else:
            # --- Simplified setup for utility use (like the LegendGenerator) ---
            # We don't have a real scene, so we center the world at origin (0,0).
            self.scene_center_world = np.array([0.0, 0.0])
            # print("ScenarioRenderer initialized in utility mode.")

        self.render_offset_pixels = np.array([self.screen_width / 2.0, self.screen_height / 2.0])

        pygame.init()
        self.surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        
        print(f"ScenarioRenderer (V2) initialized. Centered on: {self.scene_center_world}")

    def _get_default_config(self) -> Dict:
        """ --- UPDATED: Added V2 agent rendering parameters --- """
        return {
            # General
            'pixels_per_meter': 5,
            'fps': 10,
            # Colors (RGBA)
            'bg_color': (30, 30, 30, 255),
            'road_asphalt': (120, 120, 120, 255),
            'road_boundary': (80, 80, 80, 255),
            'lane_marking_white': (255, 255, 255, 255),
            'lane_marking_yellow': (255, 255, 0, 255),
            'av_color': (255, 0, 255, 255),         # <-- CHANGED TO MAGENTA
            'av_ghost_color': (255, 0, 255, 70),     # <-- NEW: Magenta (semi-transparent)
            'av_ghost_alpha': 125, # Value from 0 (transparent) to 255 (opaque)
            'vehicle_color': (0, 150, 255, 255),     # Blue
            'pedestrian_color': (255, 100, 0, 255),  # Orange
            'cyclist_color': (0, 255, 255, 255),      # Cyan
            'crosswalk_color': (200, 200, 200, 80), # Semi-transparent white
            
            # --- NEW: AV Trail Parameters ---
            'av_trail_color': (255, 0, 255),      # Same color as the AV
            'av_trail_width': 2,                  # Line thickness in pixels
            'av_trail_dash_length': 7,            # Length of a dash in pixels
            'av_trail_gap_length': 7,             # Length of the gap in pixels
            
            # --- NEW: Traffic Light Colors ---
            'tl_stop_color': (255, 0, 0, 150),       # Semi-transparent Red
            'tl_caution_color': (255, 255, 0, 150),  # Semi-transparent Yellow
            'tl_go_color': (0, 255, 0, 150),        # Semi-transparent Green
            'stop_sign_color': (255, 0, 0, 255), # Solid Red

            # Line Widths
            'asphalt_line_width': 24,
            'boundary_line_width': 2,
            'lane_line_width': 1,
            # Agent Render Sizes
            'pedestrian_radius_m': 1.0, # in meters
            'traffic_light_radius_m': 1.5, # NEW
            'stop_sign_radius_m': 0.9, # NEW

            # Waymo Type IDs (from your documentation)
            'MAP_TYPE_LANE_CENTER': {1, 2, 3}, # Note: Type 0 is often undefined lane
            'MAP_TYPE_ROAD_LINE': {11, 12, 13, 14, 15, 16, 17, 18},
            'MAP_TYPE_ROAD_EDGE': {21, 22},
            'MAP_TYPE_CROSSWALK': {41}, 
            'MAP_TYPE_STOP_SIGN': {31},

            # --- NEW: Traffic Light State IDs (from your documentation) ---
            'TL_STATE_STOP': {1, 4, 7},
            'TL_STATE_CAUTION': {2, 5, 8},
            'TL_STATE_GO': {3, 6},

            'OBJECT_TYPES': {
                'TYPE_VEHICLE': 1,
                'TYPE_PEDESTRIAN': 2,
                'TYPE_CYCLIST': 3,
            }
        }

    def _world_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        centered_coords = coords - self.scene_center_world
        pixel_coords = centered_coords * self.config['pixels_per_meter']
        pixel_coords[:, 1] *= -1
        pixel_coords += self.render_offset_pixels
        return pixel_coords.astype(int)

    def _create_lane_polygon(self, polyline: np.ndarray, width: float) -> np.ndarray:
        """
        Converts a centerline polyline into a filled polygon "ribbon".

        Args:
            polyline: An (N, 2) numpy array of points for the centerline.
            width: The desired width of the ribbon in meters.

        Returns:
            An (M, 2) numpy array of vertices for the polygon, or None if invalid.
        """
        if len(polyline) < 2:
            return None

        # Calculate the offset distance
        half_width = width / 2.0
        
        # We need to compute the perpendicular vector for each segment
        left_points = []
        right_points = []

        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i+1]
            
            # Calculate the direction vector of the segment
            direction_vec = p2 - p1
            
            # Handle potential zero-length segments
            segment_length = np.linalg.norm(direction_vec)
            if segment_length < 1e-6:
                continue
                
            direction_vec /= segment_length
            
            # The perpendicular vector is (-dy, dx)
            perp_vec = np.array([-direction_vec[1], direction_vec[0]])
            
            # Calculate the left and right offset points for this segment
            # For the first point of the segment
            left_points.append(p1 + half_width * perp_vec)
            right_points.append(p1 - half_width * perp_vec)
            
            # And for the second point of the segment
            if i == len(polyline) - 2:
                left_points.append(p2 + half_width * perp_vec)
                right_points.append(p2 - half_width * perp_vec)

        if not left_points or not right_points:
            return None

        # The final polygon is the sequence of left points followed by the
        # sequence of right points in reverse order to form a closed loop.
        # This is the key trick to creating the ribbon.
        polygon_vertices = np.vstack([left_points, right_points[::-1]])
        
        return polygon_vertices

    def _draw_regular_polygon(self, surface: pygame.Surface, center_xy_pixels: np.ndarray, radius_m: float, num_sides: int, color: Tuple[int, int, int], alpha: int = 255):
        """
        Draws a regular polygon (like a hexagon or octagon).
        """
        radius_px = int(radius_m * self.config['pixels_per_meter'])
        if radius_px < 1: return

        # Calculate the vertices of the polygon
        # We start at the "top" and go clockwise
        angle_step = 2 * np.pi / num_sides
        vertices = []
        for i in range(num_sides):
            angle = i * angle_step - (np.pi / 2) # Start at the top point
            x = center_xy_pixels[0] + radius_px * np.cos(angle)
            y = center_xy_pixels[1] + radius_px * np.sin(angle)
            vertices.append((x, y))

        # --- Use the same robust alpha-handling logic as our other shapes ---
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        shape_size = (int(max_x - min_x) + 1, int(max_y - min_y) + 1)

        if shape_size[0] < 1 or shape_size[1] < 1:
            return
            
        shape_surface = pygame.Surface(shape_size, pygame.SRCALPHA)
        
        # Draw the polygon onto the temporary surface (adjusting vertices to be local)
        pygame.draw.polygon(shape_surface, color, [(v[0] - min_x, v[1] - min_y) for v in vertices])
        
        shape_surface.set_alpha(alpha)
        
        surface.blit(shape_surface, (int(min_x), int(min_y)))
        
    def _draw_dashed_line(self, start_pos, end_pos, color, width, dash_length, gap_length):
        """
        Draws a dashed line on the main surface.
        """
        # --- FIX: Explicitly convert to float for calculation precision ---
        start_pos = np.array(start_pos, dtype=float)
        end_pos = np.array(end_pos, dtype=float)
        
        # Calculate the vector and length of the full line
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        # Avoid division by zero
        if length == 0:
            return
            
        unit_direction = direction / length
        
        current_pos = start_pos.copy()
        distance_covered = 0
        
        while distance_covered < length:
            # Calculate the end of the current dash
            dash_end = current_pos + unit_direction * dash_length
            
            # If the dash goes beyond the end point, cap it
            if distance_covered + dash_length > length:
                dash_end = end_pos
            
            # --- FIX: Convert back to integers for Pygame drawing ---
            start_pixel = (int(current_pos[0]), int(current_pos[1]))
            end_pixel = (int(dash_end[0]), int(dash_end[1]))
            
            # Draw the dash segment
            pygame.draw.line(self.surface, color, start_pixel, end_pixel, width)
            
            # Move the current position forward by the length of the dash and the gap
            current_pos += unit_direction * (dash_length + gap_length)
            distance_covered += dash_length + gap_length

# In src/rendering/scenario_renderer.py, replace the _draw_striped_polygon method

    def _draw_striped_polygon(self, surface: pygame.Surface, polygon_pixels: np.ndarray, color: Tuple[int, int, int], stripe_width: int):
        """
        --- V3: Geometric Line Drawing Method ---
        Fills a polygon with stripes by drawing parallel lines inside it.
        This is robust and geometrically correct.
        """
        if len(polygon_pixels) < 4: # Need a quadrilateral at least
            # Fallback for simple shapes
            pygame.draw.polygon(surface, color, polygon_pixels)
            return

        # 1. Find the two longest, most parallel edges. These define the direction of the crossing.
        edges = []
        for i in range(len(polygon_pixels)):
            p1 = polygon_pixels[i]
            p2 = polygon_pixels[(i + 1) % len(polygon_pixels)]
            vec = p2 - p1
            edges.append((np.linalg.norm(vec), vec / np.linalg.norm(vec), p1, p2))
        
        edges.sort(key=lambda x: x[0], reverse=True)

        # Assume the two longest edges are the sides of the crosswalk
        long_edge1_vec = edges[0][1]
        long_edge2_vec = edges[1][1]

        # Check if they are roughly parallel (dot product is close to 1 or -1)
        if np.abs(np.dot(long_edge1_vec, long_edge2_vec)) < 0.9:
             # If not parallel, the shape is too weird. Fallback to simple polygon.
            pygame.draw.polygon(surface, (180, 180, 180, 100), polygon_pixels) # Draw semi-transparent gray
            return
            
        # 2. The direction of the stripes is the average of the two *shortest* edges.
        #    This is more robust than calculating a perpendicular.
        short_edge1_vec = edges[-1][1]
        short_edge2_vec = edges[-2][1]
        stripe_dir_vector = (short_edge1_vec - short_edge2_vec) / 2.0 # Average direction, pointing inward
        stripe_length = (edges[-1][0] + edges[-2][0]) / 2.0 # Average length of a stripe

        # 3. Iterate along one of the long edges and draw the stripes.
        # We'll use the starting point of the longest edge as our anchor.
        start_point = edges[0][2]
        path_vector = edges[0][1]
        path_length = edges[0][0]
        
        num_stripes = int(path_length / (stripe_width * 2))
        if num_stripes < 1: return

        step_vector = path_vector * (path_length / num_stripes)

        for i in range(num_stripes):
            # Calculate the start and end point for this stripe
            # We add a small offset to start inside the polygon
            p1 = start_point + (i + 0.5) * step_vector
            p2 = p1 - stripe_dir_vector * stripe_length # Draw along the averaged short-edge vector
            
            # Draw the stripe as a thick line
            pygame.draw.line(surface, color, p1, p2, stripe_width)
        
        print(f'   These are the location: {polygon_pixels}')
        
    def _draw_vehicle_shape(self, center_xy_pixels: np.ndarray, width_m: float,
                            length_m: float, angle_rad: float, color: Tuple[int, int, int, int],
                            alpha: int = 255): # <-- NEW: Added optional alpha parameter
        """
        Draws a more realistic vehicle shape (a polygon with a tapered front)
        instead of a simple rectangle.
        """
        w_px = width_m * self.config['pixels_per_meter']
        l_px = length_m * self.config['pixels_per_meter']
        
        # --- THE FIX IS HERE ---
        # We must negate the angle to convert from Waymo's counter-clockwise world
        # coordinates to Pygame's clockwise screen coordinates (due to the inverted Y-axis).
        angle_rad = -angle_rad
        
        
        # 1. Define the 5 vertices of the car shape in its local coordinate system
        #    (centered at 0,0, pointing right along the x-axis).
        #    The front of the car is the "+x" direction.
        
        half_l = l_px / 2.0
        half_w = w_px / 2.0
        
        # Taper the front by a certain factor. 0.7 means the front is 70% of the back width.
        front_taper_factor = 0.25 
        
        # Vertices: [back-left, front-left, front-point, front-right, back-right]
        # We can use a simple 5-point polygon (like a house shape) to indicate the front.
        local_vertices = np.array([
            [-half_l, -half_w],  # Back-left
            [ half_l, -half_w * front_taper_factor],  # Front-left (tapered)
            [ half_l,  half_w * front_taper_factor],  # Front-right (tapered)
            [-half_l,  half_w],  # Back-right
        ])

        # 2. Create the rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        
        # 3. Rotate the local vertices and translate them to the center position
        #    The @ operator is matrix multiplication.
        rotated_vertices = local_vertices @ rotation_matrix.T
        translated_vertices = rotated_vertices + center_xy_pixels
        
        # 4. Draw the final polygon on the surface
        # --- NEW ALPHA-HANDLING LOGIC ---
        
        # 1. Create a temporary surface just for this polygon.
        #    This is necessary for applying a uniform alpha to a complex shape.
        #    We find the bounding box of the polygon to make the surface as small as possible.
        min_x, min_y = np.min(translated_vertices, axis=0)
        max_x, max_y = np.max(translated_vertices, axis=0)
        shape_size = (max_x - min_x, max_y - min_y)

        # Skip drawing if the shape is too small
        if shape_size[0] < 1 or shape_size[1] < 1:
            return
            
        shape_surface = pygame.Surface(shape_size, pygame.SRCALPHA)
        
        # 2. Draw the polygon onto this temporary surface.
        #    We need to shift the vertices to be relative to the new surface's top-left corner.
        pygame.draw.polygon(shape_surface, color, translated_vertices - [min_x, min_y])
        
        # 3. Set the overall alpha of the entire temporary surface.
        #    This is the key step that correctly handles transparency.
        shape_surface.set_alpha(alpha)
        
        # 4. Blit the transparent temporary surface onto the main surface.
        self.surface.blit(shape_surface, (min_x, min_y))

    def _draw_static_map(self):
        """ --- UPDATED: Added a rendering layer for map polygons --- """
        polylines = self.scenario_data['map_polylines']
        polyline_types = self.scenario_data['map_polyline_types']

        # --- Layer 1: The "Asphalt" Base ---
        # UPDATED: We now create and draw filled polygons for each lane.
        for i, p_type in enumerate(polyline_types):
            if p_type in self.config['MAP_TYPE_LANE_CENTER']:
                polyline_world = polylines[i][:, :2]
                
                # Use our new helper to create the ribbon polygon
                lane_polygon_world = self._create_lane_polygon(
                    polyline_world,
                    self.config['asphalt_line_width'] / self.config['pixels_per_meter'] # Convert width back to meters
                )
                
                if lane_polygon_world is not None:
                    lane_polygon_pixels = self._world_to_pixel(lane_polygon_world)
                    pygame.draw.polygon(
                        self.surface,
                        self.config['road_asphalt'],
                        points=lane_polygon_pixels
                    )
        # --- NEW: Layer 2: Map Polygons (Crosswalks, etc.) ---
        # Draw these on top of the asphalt, but under the lane lines.
        for i, p_type in enumerate(polyline_types):
            if p_type in self.config['MAP_TYPE_CROSSWALK']:
                polygon_world = polylines[i][:, :2]
                polygon_pixels = self._world_to_pixel(polygon_world)
                
                # --- THE CRITICAL CHANGE IS HERE ---
                # Call our new striped polygon function
                self._draw_striped_polygon(
                    surface=self.surface,
                    polygon_pixels=polygon_pixels,
                    color=self.config['lane_marking_white'],
                    stripe_width=4 # Just the width is needed now
                    )

        # --- NEW: Layer 3: Stop Signs ---
        # Draw these on top of the road surface so they are visible.
        for i, p_type in enumerate(polyline_types):
            if p_type in self.config['MAP_TYPE_STOP_SIGN']:
                # A stop sign is represented by a single point in the polyline
                stop_sign_world_pos = polylines[i][0, :2]
                stop_sign_pixel_pos = self._world_to_pixel(stop_sign_world_pos)[0]
                
                self._draw_regular_polygon(
                    surface=self.surface, # Pass the surface
                    center_xy_pixels=stop_sign_pixel_pos,
                    radius_m=self.config['stop_sign_radius_m'],
                    num_sides=5, # An octagon
                    color=self.config['stop_sign_color']
                )
        # --- Layer 3: Lane Markings (Unchanged, was Layer 2) ---
        for i, p_type in enumerate(polyline_types):
            if p_type in self.config['MAP_TYPE_ROAD_LINE']:
                polyline_world = polylines[i][:, :2]
                polyline_pixels = self._world_to_pixel(polyline_world)
                color = self.config['lane_marking_yellow'] if p_type >= 14 else self.config['lane_marking_white']
                if len(polyline_pixels) > 1:
                    pygame.draw.lines(self.surface, color, False, polyline_pixels, self.config['lane_line_width'])

        # --- Layer 4: Road Boundaries (Unchanged, was Layer 3) ---
        for i, p_type in enumerate(polyline_types):
            if p_type in self.config['MAP_TYPE_ROAD_EDGE']:
                polyline_world = polylines[i][:, :2]
                polyline_pixels = self._world_to_pixel(polyline_world)
                if len(polyline_pixels) > 1:
                    pygame.draw.lines(self.surface, self.config['road_boundary'], False, polyline_pixels, self.config['boundary_line_width'])
                    
                    
    def _draw_rotated_rectangle(self, center_xy_pixels: np.ndarray, width_m: float,
                                length_m: float, angle_rad: float, color: Tuple[int, int, int, int]):
        """
        Draws a rotated rectangle on the main surface.
        """
        w_px = max(1, int(width_m * self.config['pixels_per_meter']))
        l_px = max(1, int(length_m * self.config['pixels_per_meter']))
        
        rect_surface = pygame.Surface((l_px, w_px), pygame.SRCALPHA)
        rect_surface.fill(color)
        
        # --- THE FIX IS HERE: Remove the negative sign ---
        # Both Waymo's angle and pygame's rotation are counter-clockwise for positive values,
        # so we do not need to invert the angle.
        rotated_surface = pygame.transform.rotate(rect_surface, np.rad2deg(angle_rad))
        
        # Get the new bounding box of the rotated surface and set its center
        new_rect = rotated_surface.get_rect(center=tuple(center_xy_pixels))
        
        # Blit (draw) the rotated rectangle onto the main rendering surface
        self.surface.blit(rotated_surface, new_rect.topleft)

    # --- NEW: Helper method for drawing pedestrians ---
    def _draw_circle(self, center_xy_pixels: np.ndarray, radius_m: float, color: Tuple[int, int, int, int]):
        """Draws a filled circle for circular agents like pedestrians."""
        radius_px = int(radius_m * self.config['pixels_per_meter'])
        pygame.draw.circle(self.surface, color, center_xy_pixels, max(1, radius_px))


    # --- UPDATED: Main agent drawing logic now differentiates types ---
    def _draw_agents_at_timestep(self, timestep: int):
        """
        Draws all dynamic agents, rendering them with a different shape and
        color based on their object type.
        """
        traj_data = self.scenario_data['all_agent_trajectories']
        valid_mask = self.scenario_data['valid_mask']
        sdc_index = self.scenario_data['sdc_track_index']
        object_types = self.scenario_data['object_types'] # <-- Load the new data

        states_at_t = traj_data[:, timestep, :]
        valid_agents_mask = valid_mask[:, timestep]
        valid_indices = np.where(valid_agents_mask)[0]

        for agent_idx in valid_indices:
            agent_state = states_at_t[agent_idx]
            agent_type = object_types[agent_idx]
            
            center_x, center_y = agent_state[0], agent_state[1]
            center_pixels = self._world_to_pixel(np.array([center_x, center_y]))[0]

            # --- Main branching logic based on agent type ---
            if agent_idx == sdc_index:
                # --- NEW DASHED LINE TRAIL LOGIC STARTS HERE ---
                
                # 1. Get a sequence of past positions for the AV
                # We'll go back 2 seconds (20 timesteps)
                past_positions_pixels = []
                for i in range(20, -1, -5): # Look at t, t-0.5s, t-1.0s, t-1.5s, t-2.0s
                    past_ts = timestep - i
                    if past_ts >= 0 and valid_mask[agent_idx, past_ts]:
                        past_state = traj_data[agent_idx, past_ts, :]
                        past_positions_pixels.append(self._world_to_pixel(past_state[:2])[0])
                
                # 2. Draw the dashed line segments connecting these past points
                if len(past_positions_pixels) > 1:
                    for i in range(len(past_positions_pixels) - 1):
                        self._draw_dashed_line(
                            start_pos=past_positions_pixels[i],
                            end_pos=past_positions_pixels[i+1],
                            color=self.config['av_trail_color'],
                            width=self.config['av_trail_width'],
                            dash_length=self.config['av_trail_dash_length'],
                            gap_length=self.config['av_trail_gap_length']
                        )
                
                # --- TRAIL LOGIC ENDS ---
                
                # Always draw the SDC as a green rectangle
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                self._draw_vehicle_shape(center_pixels, width, length, heading, self.config['av_color'])
            
            elif agent_type == self.config['OBJECT_TYPES']['TYPE_VEHICLE']:
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                self._draw_vehicle_shape(center_pixels, width, length, heading, self.config['vehicle_color'])

            elif agent_type == self.config['OBJECT_TYPES']['TYPE_PEDESTRIAN']:
                self._draw_circle(center_pixels, self.config['pedestrian_radius_m'], self.config['pedestrian_color'])
            
            elif agent_type == self.config['OBJECT_TYPES']['TYPE_CYCLIST']:
                # Render cyclists as smaller, thinner rectangles
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                self._draw_rotated_rectangle(center_pixels, width, length, heading, self.config['cyclist_color'])

            else: # Fallback for TYPE_OTHER or TYPE_UNSET
                # Render as a default vehicle for now
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                self._draw_vehicle_shape(center_pixels, width, length, heading, self.config['vehicle_color'])

    def _draw_dynamic_map_elements(self, timestep: int):
        """
        Draws map elements whose state can change over time, like traffic lights.
        """
        dynamic_states = self.scenario_data['dynamic_map_states'][timestep, :, :]
        
        # Filter for valid traffic light states at this timestep
        # According to your docs, the first column (lane_id) is > 0 for valid lights
        valid_lights = dynamic_states[dynamic_states[:, 0] > 0]

        for light_state in valid_lights:
            state_enum = int(light_state[1])
            stop_x, stop_y = light_state[2], light_state[3]
            
            color = None
            if state_enum in self.config['TL_STATE_STOP']:
                color = self.config['tl_stop_color']
            elif state_enum in self.config['TL_STATE_CAUTION']:
                color = self.config['tl_caution_color']
            elif state_enum in self.config['TL_STATE_GO']:
                color = self.config['tl_go_color']
            
            # If the light is a recognized color, draw it
            if color:
                center_pixels = self._world_to_pixel(np.array([stop_x, stop_y]))[0]
                self._draw_circle(
                    center_xy_pixels=center_pixels,
                    radius_m=self.config['traffic_light_radius_m'],
                    color=color
                )
                
    def render_to_gif(self, output_path: str, duration_sec: float = 9.1):
        """ --- UPDATED: Now calls the dynamic map element renderer --- """
        frames = []
        num_timesteps = self.scenario_data['all_agent_trajectories'].shape[1]
        
        map_background = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.surface.fill(self.config['bg_color'])
        self._draw_static_map()
        map_background.blit(self.surface, (0, 0))

        print(f"Rendering {num_timesteps} frames to {output_path}...")
        for ts in range(num_timesteps):
            self.surface.blit(map_background, (0, 0))
            
            # --- NEW: Call the dynamic map element renderer here ---
            # We draw this before the agents so that cars appear on top of the light indicators.
            self._draw_dynamic_map_elements(ts)
            
            self._draw_agents_at_timestep(ts)
            
            frame_data = pygame.surfarray.array3d(self.surface)
            frame_data = np.rot90(frame_data, k=3)
            frame_data = np.fliplr(frame_data)
            frames.append(Image.fromarray(frame_data))

        frames[0].save(
            output_path, save_all=True, append_images=frames[1:],
            duration=1000 // self.config['fps'], loop=0
        )
        print(f"✅ GIF saved successfully to: {output_path}")
    def close(self):
        pygame.quit()
        

# --- UPDATED: Test Block using Config File ---
if __name__ == '__main__':
    print("--- Running ScenarioRenderer Test ---")
    
    # NEW: Import the config loader
    from src.utils.config_loader import load_config
    from glob import glob # To find a sample file

    try:
        # 1. Load the main project configuration
        config = load_config()
        npz_dir = config['data']['processed_npz_dir']
        print(f"Loaded data path from config: {npz_dir}")

        # 2. Find a sample scenario file to test with
        # We'll just grab the first .npz file we can find in the validation set.
        validation_dir = os.path.join(npz_dir, 'validation')
        sample_files = glob(os.path.join(validation_dir, '*.npz'))
        
        if not sample_files:
            raise FileNotFoundError(f"No .npz files found in '{validation_dir}'. Please check the path in your config.")
        
        sample_npz_path = sample_files[0]
        
        # 3. Load the scenario data using our parser
        scenario_data = load_npz_scenario(sample_npz_path)
        print(f"Successfully loaded scenario: {scenario_data['scenario_id']} from {sample_npz_path}")
        
        # 4. Initialize the renderer
        renderer = ScenarioRenderer(scenario_data)
        
        # 5. Fill the surface with the background color
        renderer.surface.fill(renderer.config['bg_color'])
        
        # 6. Call the method we want to test
        renderer._draw_static_map()
        
        # 7. Save the output as a PNG image for inspection
        output_filename = f"static_map_render_test_{scenario_data['scenario_id']}.png"
        
        frame_data = pygame.surfarray.array3d(renderer.surface)
        frame_data = np.rot90(frame_data, k=3)
        frame_data = np.fliplr(frame_data)
        img = Image.fromarray(frame_data)
        img.save(output_filename)
        
        renderer.close()
        
        print(f"\n✅ Static map render test complete.")
        print(f"   Image saved to: {os.path.abspath(output_filename)}")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"❌ Test Failed: {e}")