import os
import sys
import pygame
import numpy as np
from PIL import Image
from typing import Dict, Tuple
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull

# Add project root to path for our src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_processing.waymo_parser import load_npz_scenario # We'll use this for testing

class ScenarioRendererV6:
    """
    --- V6 Simulation-Grade Renderer ---
    Renders a Waymo scenario with a rich, unambiguous, SUMO-inspired visual language.
    This renderer consumes the structured map data from the V4 parser.
    """

    def __init__(self, scenario_data: Dict, config: Dict):
        """
        Initializes the renderer.

        Args:
            scenario_data: The dictionary loaded by the V4 `waymo_parser`.
            config: A dictionary containing rendering configurations (colors, widths, etc.).
        """
        self.scenario_data = scenario_data
        self.config = config
        self.map_layers = scenario_data['map_layers']

        # --- Setup the rendering canvas ---
        self.screen_width = 896
        self.screen_height = 896
        
        # Determine the camera center (same logic as before)
        sdc_index = self.scenario_data['sdc_track_index']
        sdc_traj = self.scenario_data['all_agent_trajectories'][sdc_index]
        sdc_valid_mask = self.scenario_data['valid_mask'][sdc_index]
        first_valid_idx = np.where(sdc_valid_mask)[0]
        
        if len(first_valid_idx) == 0:
            raise ValueError("SDC has no valid timesteps.")
        
        # We use the 40th valid step (or start) as the anchor
        anchor_idx = first_valid_idx[min(40, len(first_valid_idx)-1)]
        
        self.scene_center_world = sdc_traj[anchor_idx, :2]
        # --- NEW: Calculate Rotation Angle ---
        # We want the SDC's heading to point UP (which is +Y in math, -Y in pygame).
        # Standard Math: 0=Right, 90=Up.
        # We want to rotate the world so SDC_heading becomes 90 degrees (pi/2).
        # Rotation = Target - Current
        sdc_heading_at_start = sdc_traj[anchor_idx, 6]
        self.rotation_angle = (np.pi / 2.0) - sdc_heading_at_start
        
        # Pre-calculate sine/cosine for efficiency
        self.cos_rot = np.cos(self.rotation_angle)
        self.sin_rot = np.sin(self.rotation_angle)

        self.render_offset_pixels = np.array([self.screen_width / 2.0, self.screen_height / 2.0])
        
        pygame.init()
        
        # --- NEW: Initialize font for stop signs ---
        try:
            self.stop_sign_font = pygame.font.SysFont("dejavusans", 8, bold=True)
        except pygame.error:
            print("Warning: Default font 'dejavusans' not found for stop sign text.")
            self.stop_sign_font = pygame.font.Font(None, 8) # Fallback to default
            
        # Note: We will be creating surfaces dynamically, so we don't need a main 'self.surface'
        print("ScenarioRendererV6 initialized.")
    
    # --- Top-Level Orchestrator ---

    def render_to_gif(self, output_path: str):
        """
        The main public method. Renders the full scenario to an animated GIF.
        """
        print(f"Starting V6 render for scenario {self.scenario_data['scenario_id']}...")
        frames = []
        num_timesteps = self.scenario_data['all_agent_trajectories'].shape[1]

        # 1. Create the static 'map_background' surface ONCE. This is a major optimization.
        print("Step 1/3: Creating static map background...")
        map_background = self._create_static_map_surface()
        print(" -> Static map background created.")

        # 2. Loop through each timestep to render the dynamic elements.
        print(f"Step 2/3: Rendering {num_timesteps} dynamic frames...")
        for ts in range(num_timesteps):
            # 3. Create a fresh surface for this frame.
            frame_surface = pygame.Surface((self.screen_width, self.screen_height))
            
            # 4. Blit the pre-rendered static map onto the frame.
            frame_surface.blit(map_background, (0, 0))

            # 5. Draw the dynamic elements for this timestep ON TOP.
            self._draw_dynamic_layer(frame_surface, ts)
            
            # 6. Capture the completed frame_surface and convert for PIL.
            frame_data = pygame.surfarray.array3d(frame_surface)
            frame_data = np.rot90(frame_data, k=3)
            frame_data = np.fliplr(frame_data)
            frames.append(Image.fromarray(frame_data))

        # 7. Save the list of frames as a GIF.
        print("Step 3/3: Saving GIF...")
        frames[0].save(
            output_path, save_all=True, append_images=frames[1:],
            duration=1000 // self.config['fps'], loop=0
        )
        print(f"✅ GIF saved successfully to: {output_path}")

    # --- Helper Methods (Stubs for now) ---
    
    def _create_static_map_surface(self) -> pygame.Surface:
        """
        Creates a single surface containing all the static map elements,
        drawn in the correct, layered order.
        """
        static_surface = pygame.Surface((self.screen_width, self.screen_height))
        
        # Layer 0: Base Layer
        static_surface.fill(self.config['colors']['background_verge'])

        # Layer 1: Surfaces
        self._draw_surfaces(static_surface)

        # Layer 2: Markings
        self._draw_markings(static_surface)
        
        # Layer 3: Static Objects
        self._draw_static_objects(static_surface)

        # --- NEW: Layer 4: Boundaries ---
        # Draw these last to create a crisp outline over all surfaces.
        self._draw_boundaries(static_surface)

        return static_surface

    def _draw_surfaces(self, surface: pygame.Surface):
        """
        Draws all the "ground" polygons in the correct order.
        - Driveways and sidewalks are drawn first.
        - Road ribbons are drawn on top.
        """
        print("   -> Drawing surfaces...")
        
        # --- Layer 1.1: Driveways (if they exist) ---
        # These are usually at the edge, so we draw them first.
        driveway_polys = self.map_layers.get('driveways', [])
        for poly_world in driveway_polys:
            if poly_world.shape[0] < 3: continue
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            pygame.draw.polygon(surface, self.config['colors']['surface_driveway'], poly_pixels)

        # Note: Waymo data doesn't typically provide sidewalk polygons,
        # so we will rely on road edges to define their boundary.

        # --- Layer 1.2: Road Surfaces (as wide ribbons) ---
        # --- Layer 1.2: Road & Bike Lane Surfaces ---
        
        road_width_m = self.config['widths']['road_lane_width_m']
        bike_lane_width_m = self.config['widths'].get('bike_lane_width_m', 1.5)
        
        # Create a single list of all lane types to iterate through
        all_lanes = [
            (self.map_layers.get('freeway_lanes', []), self.config['colors']['surface_road'], road_width_m),
            (self.map_layers.get('surface_street_lanes', []), self.config['colors']['surface_road'], road_width_m),
            (self.map_layers.get('bike_lanes', []), self.config['colors']['surface_bike_lane'], bike_lane_width_m),
        ]

        for lanes, color, width_m in all_lanes:
            for centerline_world in lanes:
                # --- The CRITICAL FIX ---
                # We will draw a slightly wider ribbon than the nominal lane width.
                # This ensures that adjacent ribbons overlap properly, creating a
                # solid, gap-free surface without incorrectly filling non-drivable areas.
                # An expansion of 15-20% is a good heuristic.
                expanded_width_m = width_m * 1.5 
                
                ribbon_poly_world = self._create_lane_polygon(centerline_world, expanded_width_m)
                
                if ribbon_poly_world is not None:
                    ribbon_poly_pixels = self._world_to_pixel(ribbon_poly_world)
                    pygame.draw.polygon(surface, color, ribbon_poly_pixels)
                        


    def _draw_traffic_light_bar(self, surface: pygame.Surface, stop_point_world: np.ndarray,
                                orientation_vec: np.ndarray, color: Tuple[int, int, int]):
        """ --- FINAL ROBUST VERSION --- """
        # The orientation_vec IS the perpendicular vector. Ensure it's a unit vector.
        norm = np.linalg.norm(orientation_vec)
        if norm < 1e-6: return
        unit_perp_vec = orientation_vec / norm

        # Dimensions
        bar_length_m = self.config['widths']['road_lane_width_m'] * 1.2 # Make it a bit wider than the lane
        bar_thickness_px = self.config['widths']['traffic_light_bar_px']
        half_bar_length_m = bar_length_m / 2.0

        # Calculate endpoints in world coordinates
        p1_world = stop_point_world + unit_perp_vec * half_bar_length_m
        p2_world = stop_point_world - unit_perp_vec * half_bar_length_m

        # Convert to pixels and draw a simple, thick line
        p1_pixels = self._world_to_pixel(p1_world)[0]
        p2_pixels = self._world_to_pixel(p2_world)[0]
        pygame.draw.line(surface, color, p1_pixels, p2_pixels, bar_thickness_px)
        
        # Print a debug information. As much as posible
        # print(f"   -> Drew traffic light bar for lane {lane_id} at {stop_point_world}.")
        # print(f"       Bar endpoints (world): {bar_p1_world} to {bar_p2_world}")
        # print(f"       Bar endpoints (pixels): {bar_p1_pixels} to {bar_p2_pixels}")
        # print(f"       Lane direction vector: {direction_vec}, Perpendicular vector: {perp_vec}")
        
    def _draw_markings(self, surface: pygame.Surface):
        """
        --- FINAL VERSION with Centerlines ---
        Draws all "paint" on top of the road surfaces, including centerlines,
        crosswalks, and all types of lane boundary lines.
        """
        print("   -> Drawing markings...")
        
        white = self.config['colors']['marking_white']
        yellow = self.config['colors']['marking_yellow']
        centerline_color = self.config['colors']['marking_centerline']
        solid_width = self.config['widths']['line_solid_px']
        dashed_width = self.config['widths']['line_dashed_px']

        # --- NEW: Layer 2.0: Lane Centerlines ---
        # Draw these first as faint guides.
        all_centerlines = (
            self.map_layers.get('surface_street_lanes', []) +
            self.map_layers.get('freeway_lanes', []) +
            self.map_layers.get('bike_lanes', [])
        )
        for poly_world in all_centerlines:
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            if len(poly_pixels) > 1:
                pygame.draw.lines(surface, centerline_color, False, poly_pixels, 1) # Draw as thin, 1px lines

        # --- Layer 2.1: Crosswalks ---
        for poly_world in self.map_layers.get('crosswalks', []):
            self._draw_striped_polygon(surface, poly_world, white, stripe_width=4)

        # --- Layer 2.2: Lane Boundary Lines ---
        for poly_world in self.map_layers.get('lines_white_dashed', []):
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            if len(poly_pixels) < 2: continue
            for i in range(len(poly_pixels) - 1):
                self._draw_dashed_line(surface, poly_pixels[i], poly_pixels[i+1], white, dashed_width, 7, 7)

        for poly_world in self.map_layers.get('lines_white_solid_single', []):
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            if len(poly_pixels) > 1:
                pygame.draw.lines(surface, white, False, poly_pixels, solid_width)
        for poly_world in self.map_layers.get('lines_white_solid_double', []):
            self._draw_double_line(surface, poly_world, white, solid_width, gap_m=0.3)
        
        # ... (rest of the yellow line logic remains the same) ...
        for poly_world in self.map_layers.get('lines_yellow_dashed', []):
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            if len(poly_pixels) < 2: continue
            for i in range(len(poly_pixels) - 1):
                self._draw_dashed_line(surface, poly_pixels[i], poly_pixels[i+1], yellow, dashed_width, 7, 7)
        for poly_world in self.map_layers.get('lines_yellow_broken_double', []):
             self._draw_double_line(surface, poly_world, yellow, dashed_width, gap_m=0.3)
        for poly_world in self.map_layers.get('lines_yellow_solid_single', []):
             poly_pixels = self._world_to_pixel(poly_world[:, :2])
             if len(poly_pixels) > 1:
                pygame.draw.lines(surface, yellow, False, poly_pixels, solid_width)
        for poly_world in self.map_layers.get('lines_yellow_solid_double', []):
            self._draw_double_line(surface, poly_world, yellow, solid_width, gap_m=0.3)
        for poly_world in self.map_layers.get('lines_yellow_passing_double', []):
             self._draw_double_line(surface, poly_world, yellow, solid_width, gap_m=0.3)

    def _draw_static_objects(self, surface: pygame.Surface):
        """
        Draws static map objects like stop signs, now with text.
        """
        print("   -> Drawing static objects...")
        
        stop_sign_color = self.config['colors']['stop_sign']
        stop_sign_radius_m = self.config['radii']['stop_sign_m']
        text_color = self.config['colors']['marking_white'] # White text

        for stop_sign_pos_world in self.map_layers['stop_signs']:
            stop_sign_pos_pixels = self._world_to_pixel(stop_sign_pos_world[:2])[0]
            
            # 1. Draw the red pentagon background (same as before)
            self._draw_regular_polygon(
                surface=surface,
                center_xy_pixels=stop_sign_pos_pixels,
                radius_m=stop_sign_radius_m,
                num_sides=5,
                color=stop_sign_color
            )
            
            # 2. Render the "STOP" text
            text_surface = self.stop_sign_font.render("STOP", True, text_color)
            
            # 3. Calculate the position to center the text on the sign
            text_rect = text_surface.get_rect(center=tuple(stop_sign_pos_pixels))
            
            # 4. Blit the text on top of the pentagon
            surface.blit(text_surface, text_rect)


    def _draw_dynamic_layer(self, surface: pygame.Surface, timestep: int):
        """
        Draws all dynamic elements for a single timestep on top of the provided surface.
        This includes traffic light states and all moving agents.
        """
        
        # --- Part 1: Draw Dynamic Map Elements (Traffic Lights) ---
        if 'dynamic_map_states' in self.scenario_data:
            dynamic_states = self.scenario_data['dynamic_map_states'][timestep, :, :]
            valid_lights = dynamic_states[dynamic_states[:, 0] > 0]
            
            # --- NEW: Group lights by their unique stop point location ---
            lights_by_location = {}
            for light in valid_lights:
                location_key = (light[2], light[3]) # (stop_x, stop_y)
                if location_key not in lights_by_location:
                    lights_by_location[location_key] = []
                lights_by_location[location_key].append(light)

            # --- Now, process each unique location ---
            for location, lights_in_group in lights_by_location.items():
                stop_point_world = np.array(location)
                
                # print(f"\n--- DEBUG: Processing Light Group at {stop_point_world} at timestep {timestep}---")
                
                # a) Determine the most restrictive state for the group
                states = [int(l[1]) for l in lights_in_group]
                final_state = 0
                if any(s in {1, 4, 7} for s in states): final_state = 1 # STOP
                elif any(s in {2, 5, 8} for s in states): final_state = 2 # CAUTION
                elif any(s in {3, 6} for s in states): final_state = 3 # GO
                
                # b) Determine the best orientation by averaging lane vectors
                orientation_vectors = []
                for light in lights_in_group:
                    lane_id = int(light[0])
                    lane_polyline = self.scenario_data['lane_id_to_polyline'].get(lane_id)
                    
                    if lane_polyline is not None and len(lane_polyline) >= 2:
                        # 1. Compute distance from each polyline point to stop point
                        diffs = lane_polyline[:, :2] - stop_point_world
                        dists = np.linalg.norm(diffs, axis=1)

                        # 2. Find closest index
                        idx = int(np.argmin(dists))

                        # 3. Select neighbor points for tangent direction
                        if 0 < idx < len(lane_polyline) - 1:
                            p_prev = lane_polyline[idx - 1, :2]
                            p_next = lane_polyline[idx + 1, :2]
                            tangent = p_next - p_prev
                        elif idx == 0:
                            p_next = lane_polyline[idx + 1, :2]
                            tangent = p_next - lane_polyline[idx, :2]
                        else:  # idx == len - 1
                            p_prev = lane_polyline[idx - 1, :2]
                            tangent = lane_polyline[idx, :2] - p_prev

                        # 4. Normalize tangent
                        norm = np.linalg.norm(tangent)
                        if norm < 1e-6:
                            # Skip this lane if degenerate
                            continue
                        tangent = tangent / norm

                        # 5. Compute perpendicular
                        perp_vec = np.array([-tangent[1], tangent[0]])

                        # 6. Normalize perpendicular
                        perp_vec = perp_vec / np.linalg.norm(perp_vec)

                        # 7. Add to orientation vectors list
                        orientation_vectors.append(perp_vec)
                        
                        # print(f"  - Lane ID: {lane_id}")
                        # print(f"    - Direction Vec: [{direction_vec[0]:.2f}, {direction_vec[1]:.2f}]")
                        # print(f"    - Normalized Perp Vec: [{norm_perp_vec[0]:.2f}, {norm_perp_vec[1]:.2f}]")
                
                if not orientation_vectors:
                    print("  - No valid orientation vectors found for this group. Skipping.")
                    continue
                
                # --- This is the calculation we need to inspect ---
                avg_orientation_vec = np.mean(orientation_vectors, axis=0)
                # print(f"  - Vector Count: {len(orientation_vectors)}")
                # print(f"  - AVERAGE Orientation Vec (Before Normalization): [{avg_orientation_vec[0]:.2f}, {avg_orientation_vec[1]:.2f}]")

                # Normalize the final average vector to be safe
                final_orientation_len = np.linalg.norm(avg_orientation_vec)
                if final_orientation_len < 1e-6:
                    print("  - ❗️ WARNING: Average vector is near-zero. Using first vector as fallback.")
                    # Fallback to just using the first available orientation
                    avg_orientation_vec = orientation_vectors[0]
                else:
                    avg_orientation_vec = avg_orientation_vec / final_orientation_len

                # print(f"  - FINAL Orientation Vec (Normalized): [{avg_orientation_vec[0]:.2f}, {avg_orientation_vec[1]:.2f}]")
                # print("--- END DEBUG BLOCK ---")
                
                # c) Get the color and draw the bar ONCE
                color_tuple = None
                if final_state == 1: color_tuple = self.config['colors']['tl_stop'][:3]
                elif final_state == 2: color_tuple = self.config['colors']['tl_caution'][:3]
                elif final_state == 3: color_tuple = self.config['colors']['tl_go'][:3]

                if color_tuple:
                    rot_x = avg_orientation_vec[0] * self.cos_rot - avg_orientation_vec[1] * self.sin_rot
                    rot_y = avg_orientation_vec[0] * self.sin_rot + avg_orientation_vec[1] * self.cos_rot
                    rotated_orientation = np.array([rot_x, rot_y])
                    
                    # print(f'Changing the rotation from {avg_orientation_vec} to {rotated_orientation}')
            
                    self._draw_traffic_light_bar(
                        surface=surface,
                        stop_point_world=stop_point_world,
                        orientation_vec=avg_orientation_vec,
                        color=color_tuple
                    )
                                    
        # --- Part 2: Draw All Agents ---
        
        traj_data = self.scenario_data['all_agent_trajectories']
        valid_mask = self.scenario_data['valid_mask']
        sdc_index = self.scenario_data['sdc_track_index']
        object_types = self.scenario_data['object_types']

        states_at_t = traj_data[:, timestep, :]
        valid_agents_mask = valid_mask[:, timestep]
        valid_indices = np.where(valid_agents_mask)[0]

        for agent_idx in valid_indices:
            agent_state = states_at_t[agent_idx]
            agent_type = object_types[agent_idx]
            center_pixels = self._world_to_pixel(agent_state[:2])[0]

            if agent_idx == sdc_index:
                # --- Draw the AV's Dashed Line Trail ---
                past_positions_pixels = []
                for i in range(20, -1, -5):
                    past_ts = timestep - i
                    if past_ts >= 0 and valid_mask[agent_idx, past_ts]:
                        past_state = traj_data[agent_idx, past_ts, :]
                        past_positions_pixels.append(self._world_to_pixel(past_state[:2])[0])
                
                if len(past_positions_pixels) > 1:
                    for i in range(len(past_positions_pixels) - 1):
                        self._draw_dashed_line(
                            surface=surface,
                            start_pos=past_positions_pixels[i],
                            end_pos=past_positions_pixels[i+1],
                            color=self.config['colors']['av'],
                            width=self.config['widths']['line_dashed_px'],
                            dash_len=7, gap_len=7
                        )

                # --- Draw the main AV shape on top ---
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                
                raw_heading = agent_state[6]
                corrected_heading = raw_heading + self.rotation_angle
                 
                self._draw_sumo_vehicle_shape(surface, center_pixels, width, length, corrected_heading, self.config['colors']['av'])            
            elif agent_type == 1: # TYPE_VEHICLE
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                raw_heading = agent_state[6]
                corrected_heading = raw_heading + self.rotation_angle
                self._draw_sumo_vehicle_shape(surface, center_pixels, width, length, corrected_heading, self.config['colors']['vehicle'])
                 
            elif agent_type == 2: # TYPE_PEDESTRIAN
                # print(f"   -> Drawing pedestrian at {agent_state[:2]}")                
                raw_heading = agent_state[6]
                corrected_heading = raw_heading + self.rotation_angle
                
                self._draw_sumo_pedestrian(
                    surface=surface,
                    center_xy_pixels=center_pixels,
                    radius_m=self.config['radii']['pedestrian_m'],
                    heading_rad=corrected_heading,
                    body_color=self.config['colors']['pedestrian_body'] # Just need one color
                )
            
            elif agent_type == 3: # TYPE_CYCLIST
                length, width, heading = agent_state[3], agent_state[4], agent_state[6]
                
                raw_heading = agent_state[6]
                corrected_heading = raw_heading + self.rotation_angle
                                
                # print(f"   -> Drawing cyclist at {agent_state[:2]}")
                self._draw_cyclist_shape(
                    surface=surface,
                    center_xy_pixels=center_pixels,
                    length_m=length, # The main dimension is length
                    heading_rad=corrected_heading,
                    body_color=self.config['colors']['cyclist_body'],
                    wheel_color=self.config['colors']['bicycle_wheels']
                )
            elif agent_type == 4: # TYPE_OTHER
                print(f"   -> Drawing other agent at {agent_state[:2]}")
                self._draw_regular_polygon(surface, center_pixels, 1.0, 4, self.config['colors']['other'])

    def _create_lane_polygon(self, centerline: np.ndarray, width_m: float) -> np.ndarray:
        """
        Converts a centerline polyline into a filled polygon "ribbon" of a given width.
        """
        if len(centerline) < 2:
            return None

        # This logic is complex but correct. We'll treat it as a black box for now.
        half_width = width_m / 2.0
        left_points, right_points = [], []

        for i in range(len(centerline) - 1):
            p1, p2 = centerline[i, :2], centerline[i+1, :2]
            direction_vec = p2 - p1
            segment_length = np.linalg.norm(direction_vec)
            if segment_length < 1e-6: continue
            
            direction_vec /= segment_length
            perp_vec = np.array([-direction_vec[1], direction_vec[0]])
            
            left_points.append(p1 + half_width * perp_vec)
            right_points.append(p1 - half_width * perp_vec)
            
            if i == len(centerline) - 2:
                left_points.append(p2 + half_width * perp_vec)
                right_points.append(p2 - half_width * perp_vec)

        if not left_points or not right_points:
            return None

        return np.vstack([left_points, right_points[::-1]])
    
    def _draw_dashed_line(self, surface: pygame.Surface, start_pos, end_pos, color, width, dash_len, gap_len):
        # This is our existing, robust dashed line function
        start_pos, end_pos = np.array(start_pos, dtype=float), np.array(end_pos, dtype=float)
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        if length == 0: return
        unit_dir = direction / length
        
        current_pos = start_pos.copy()
        dist_covered = 0
        while dist_covered < length:
            dash_end = current_pos + unit_dir * dash_len
            if dist_covered + dash_len > length:
                dash_end = end_pos
            
            start_pixel = (int(current_pos[0]), int(current_pos[1]))
            end_pixel = (int(dash_end[0]), int(dash_end[1]))
            pygame.draw.line(surface, color, start_pixel, end_pixel, width)
            
            current_pos += unit_dir * (dash_len + gap_len)
            dist_covered += dash_len + gap_len


    def _draw_double_line(self, surface: pygame.Surface, polyline_world: np.ndarray, color: Tuple[int, int, int], width_px: int, gap_m: float):
        """
        --- CORRECTED VERSION ---
        Draws two solid lines parallel to a given centerline polyline.
        """
        if len(polyline_world) < 2: return
        
        half_gap_m = gap_m / 2.0
        points_left, points_right = [], []
        
        # All calculations are done in world coordinates first
        for i in range(len(polyline_world) - 1):
            p1, p2 = polyline_world[i, :2], polyline_world[i+1, :2]
            
            direction_vec = p2 - p1
            segment_length = np.linalg.norm(direction_vec)
            if segment_length < 1e-6: continue
            
            perp_vec = np.array([-direction_vec[1], direction_vec[0]]) / segment_length
            
            points_left.append(p1 + half_gap_m * perp_vec)
            points_right.append(p1 - half_gap_m * perp_vec)
            if i == len(polyline_world) - 2: # Add the last point
                points_left.append(p2 + half_gap_m * perp_vec)
                points_right.append(p2 - half_gap_m * perp_vec)

        if not points_left or not points_right: return

        # Convert to pixel coordinates ONLY at the end, before drawing
        pygame.draw.lines(surface, color, False, self._world_to_pixel(np.array(points_left)), width_px)
        pygame.draw.lines(surface, color, False, self._world_to_pixel(np.array(points_right)), width_px)

    def _draw_striped_polygon(self, surface: pygame.Surface, polygon_world: np.ndarray, color: Tuple[int, int, int], stripe_width: int):
        # This is our robust V3 geometric line drawing method
        polygon_pixels = self._world_to_pixel(polygon_world[:, :2])
        if len(polygon_pixels) < 4:
            pygame.draw.polygon(surface, (180, 180, 180, 100), polygon_pixels)
            return

        edges = []
        for i in range(len(polygon_pixels)):
            p1, p2 = polygon_pixels[i], polygon_pixels[(i + 1) % len(polygon_pixels)]
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length == 0: continue
            edges.append((length, vec / length, p1, p2))
        
        edges.sort(key=lambda x: x[0], reverse=True)
        if len(edges) < 4: return

        long_edge1_vec, long_edge2_vec = edges[0][1], edges[1][1]
        if np.abs(np.dot(long_edge1_vec, long_edge2_vec)) < 0.9:
            pygame.draw.polygon(surface, (180, 180, 180, 100), polygon_pixels)
            return
            
        short_edge1_vec, short_edge2_vec = edges[-1][1], edges[-2][1]
        stripe_dir_vector = (short_edge1_vec - short_edge2_vec) / 2.0
        stripe_length = (edges[-1][0] + edges[-2][0]) / 2.0

        start_point, path_vector, path_length = edges[0][2], edges[0][1], edges[0][0]
        num_stripes = int(path_length / (stripe_width * 2))
        if num_stripes < 1: return

        step_vector = path_vector * (path_length / num_stripes)
        for i in range(num_stripes):
            p1 = start_point + (i + 0.5) * step_vector
            p2 = p1 - stripe_dir_vector * stripe_length
            pygame.draw.line(surface, color, p1, p2, stripe_width)
            
        
    def _draw_boundaries(self, surface: pygame.Surface):
        """
        Draws the physical boundaries of the road (curbs and medians)
        to give the road a crisp edge.
        """
        print("   -> Drawing boundaries...")
        
        # We can use the same color for both edges and medians for now
        boundary_color = self.config['colors']['marking_white'] # Or a custom gray
        boundary_width = self.config['widths']['line_solid_px']

        for poly_world in self.map_layers['road_edges']:
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            if len(poly_pixels) > 1:
                pygame.draw.lines(surface, boundary_color, False, poly_pixels, boundary_width)
                
        for poly_world in self.map_layers['medians']:
            poly_pixels = self._world_to_pixel(poly_world[:, :2])
            if len(poly_pixels) > 1:
                pygame.draw.lines(surface, boundary_color, False, poly_pixels, boundary_width)
                
    def _draw_regular_polygon(self, surface: pygame.Surface, center_xy_pixels: np.ndarray, radius_m: float, num_sides: int, color: Tuple[int, int, int], alpha: int = 255):
        """
        Draws a regular polygon (like a pentagon for stop signs) with alpha support.
        """
        radius_px = int(radius_m * self.config['pixels_per_meter'])
        if radius_px < 2: return # Don't draw if too small to see

        # Calculate vertices
        angle_step = 2 * np.pi / num_sides
        vertices = []
        for i in range(num_sides):
            angle = i * angle_step - (np.pi / 2) # Start at top
            x = center_xy_pixels[0] + radius_px * np.cos(angle)
            y = center_xy_pixels[1] + radius_px * np.sin(angle)
            vertices.append((x, y))

        # Use a temporary surface for robust alpha blending
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        shape_size = (int(max_x - min_x) + 1, int(max_y - min_y) + 1)
        if shape_size[0] < 1 or shape_size[1] < 1: return
            
        shape_surface = pygame.Surface(shape_size, pygame.SRCALPHA)
        pygame.draw.polygon(shape_surface, color, [(v[0] - min_x, v[1] - min_y) for v in vertices])
        
        shape_surface.set_alpha(alpha)
        surface.blit(shape_surface, (int(min_x), int(min_y)))

    def _draw_circle(self, surface: pygame.Surface, center_xy_pixels: np.ndarray, radius_m: float, color: Tuple[int, int, int], alpha: int = 255):
        """
        Draws a circle with alpha support.
        """
        radius_px = int(radius_m * self.config['pixels_per_meter'])
        if radius_px < 1: return

        # Create a temporary surface for robust alpha blending
        # The surface needs to be 2x the radius
        temp_surface_size = (radius_px * 2, radius_px * 2)
        temp_surface = pygame.Surface(temp_surface_size, pygame.SRCALPHA)

        # Draw the circle in the center of the temporary surface
        pygame.draw.circle(temp_surface, color, (radius_px, radius_px), radius_px)
        
        # Set the alpha for the entire surface
        temp_surface.set_alpha(alpha)
        
        # Blit the transparent circle onto the main surface
        blit_position = (center_xy_pixels[0] - radius_px, center_xy_pixels[1] - radius_px)
        surface.blit(temp_surface, blit_position)
        
    def _draw_sumo_vehicle_shape(self, surface: pygame.Surface, center_xy_pixels: np.ndarray, width_m: float,
                                 length_m: float, angle_rad: float, color: Tuple[int, int, int],
                                 alpha: int = 255):
        """
        --- V2: Multi-Layer SUMO Shape ---
        Draws a high-fidelity, SUMO-style vehicle shape using two polygons:
        a base for the body and a top layer for the "glass".
        """
        # Apply the rotation fix
        angle_rad = -angle_rad
        
        l_px = max(1, length_m * self.config['pixels_per_meter'])
        w_px = max(1, width_m * self.config['pixels_per_meter'])
        
        # --- 1. Define Vertices for BOTH Polygons (Body and Glass) ---
        half_l, half_w = l_px / 2.0, w_px / 2.0
        
        body_w_f = 1.0
        glass_w_f = 0.8
        hood_taper_f = 0.8
        glass_start_x = -0.6 * half_l
        glass_end_x = 0.2 * half_l
        
        # 8 vertices for the main car body
        body_vertices = np.array([
            [-half_l, -half_w * body_w_f], [glass_start_x, -half_w * body_w_f],
            [glass_end_x, -half_w * glass_w_f], [half_l, -half_w * hood_taper_f],
            [half_l, half_w * hood_taper_f], [glass_end_x, half_w * glass_w_f],
            [glass_start_x, half_w * body_w_f], [-half_l, half_w * body_w_f],
        ])

        # 4 vertices for the "glasshouse" on top
        glass_vertices = np.array([
            [glass_start_x, -half_w * glass_w_f],
            [glass_end_x, -half_w * glass_w_f],
            [glass_end_x, half_w * glass_w_f],
            [glass_start_x, half_w * glass_w_f],
        ])
        
        # --- 2. Rotate and Translate All Vertices ---
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        rotated_body_verts = body_vertices @ rot_matrix.T
        translated_body_verts = rotated_body_verts + center_xy_pixels
        
        rotated_glass_verts = glass_vertices @ rot_matrix.T
        translated_glass_verts = rotated_glass_verts + center_xy_pixels

        # --- 3. Draw the Polygons in the Correct Order (Body first, then Glass) ---
        
        # Draw the main body (using the robust alpha-blending technique)
        self._draw_polygon_alpha(surface, translated_body_verts, color, alpha)
        
        # Draw the glass on top with a darker color and its own alpha
        glass_color = self.config['colors']['vehicle_glass']
        glass_alpha = int(alpha * 0.8) # Make glass slightly more transparent
        self._draw_polygon_alpha(surface, translated_glass_verts, glass_color, glass_alpha)

    def _draw_polygon_alpha(self, surface: pygame.Surface, vertices: np.ndarray, color: Tuple[int, int, int], alpha: int = 255):
        """
        A new helper method to robustly draw any polygon with a uniform alpha.
        """
        if len(vertices) < 3: return
        
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        shape_size = (int(max_x - min_x) + 1, int(max_y - min_y) + 1)
        if shape_size[0] < 1 or shape_size[1] < 1: return
            
        shape_surface = pygame.Surface(shape_size, pygame.SRCALPHA)
        pygame.draw.polygon(shape_surface, color, vertices - [min_x, min_y])
        
        shape_surface.set_alpha(alpha)
        surface.blit(shape_surface, (int(min_x), int(min_y)))
        

    def _draw_vehicle_shape(self, surface: pygame.Surface, center_xy_pixels: np.ndarray, width_m: float,
                            length_m: float, angle_rad: float, color: Tuple[int, int, int],
                            alpha: int = 255):
        """
        Draws a vehicle shape polygon with alpha support.
        """
        w_px = max(1, width_m * self.config['pixels_per_meter'])
        l_px = max(1, length_m * self.config['pixels_per_meter'])
        
        angle_rad = -angle_rad
        
        # Define vertices
        half_l, half_w, taper = l_px / 2.0, w_px / 2.0, 0.7
        local_vertices = np.array([
            [-half_l, -half_w], [half_l, -half_w * taper],
            [half_l, half_w * taper], [-half_l, half_w]
        ])
        
        # Rotate and translate
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = local_vertices @ rot_matrix.T
        translated_vertices = rotated_vertices + center_xy_pixels
        
        # Use the same robust temporary surface method for alpha blending
        min_x, min_y = np.min(translated_vertices, axis=0)
        max_x, max_y = np.max(translated_vertices, axis=0)
        shape_size = (int(max_x - min_x) + 1, int(max_y - min_y) + 1)
        if shape_size[0] < 1 or shape_size[1] < 1: return
            
        shape_surface = pygame.Surface(shape_size, pygame.SRCALPHA)
        pygame.draw.polygon(shape_surface, color, translated_vertices - [min_x, min_y])
        
        shape_surface.set_alpha(alpha)
        surface.blit(shape_surface, (int(min_x), int(min_y)))


    def _draw_pedestrian_shape(self, surface: pygame.Surface, center_xy_pixels: np.ndarray,
                               radius_m: float, heading_rad: float, body_color: Tuple[int, int, int],
                               head_color: Tuple[int, int, int], alpha: int = 255):
        """
        Draws a SUMO-style pedestrian icon with a body and a head.
        The head is offset in the direction of the pedestrian's heading.
        """
        body_radius_px = int(radius_m * self.config['pixels_per_meter'])
        head_radius_px = int(body_radius_px * 0.6) # Head is 60% of the body radius
        if body_radius_px < 2: return

        # --- 1. Draw the Body (as a circle) ---
        self._draw_circle(
            surface=surface,
            center_xy_pixels=center_xy_pixels,
            radius_m=radius_m,
            color=body_color,
            alpha=alpha
        )

        # --- 2. Calculate the Head Position ---
        # We need to offset the head from the body center in the direction of travel.
        # Use the same Y-inversion fix for the angle.
        corrected_heading = -heading_rad
        offset_distance = body_radius_px * 0.3 # Offset the head slightly forward
        
        head_offset_x = offset_distance * np.cos(corrected_heading)
        head_offset_y = offset_distance * np.sin(corrected_heading)
        
        head_center_pixels = center_xy_pixels + np.array([head_offset_x, head_offset_y])

        # --- 3. Draw the Head (as a smaller, darker circle) ---
        # To make it a circle in Pygame, the radius needs to be an integer.
        pygame.draw.circle(
            surface,
            head_color,
            (int(head_center_pixels[0]), int(head_center_pixels[1])),
            head_radius_px
        )


# Add this new method to the ScenarioRendererV6 class

    def _draw_sumo_pedestrian(self, surface: pygame.Surface, center_xy_pixels: np.ndarray,
                              radius_m: float, heading_rad: float, body_color: Tuple[int, int, int],
                              alpha: int = 255):
        """
        --- UPDATED: SYMBOLIC SCALING ---
        Enforces a minimum pixel size so Pedestrians are always visible to the VLM.
        """
        # 1. Calculate Dimensions with a Floor
        # Physical size OR Minimum 9px radius (18px diameter)
        body_radius_px = max(9, int(radius_m * self.config['pixels_per_meter']))
        
        # 2. Shoulder Ellipses
        shoulder_rect = pygame.Rect(0, 0, body_radius_px * 2, body_radius_px * 1.2)
        
        # 3. Create Temp Surface
        canvas_size = body_radius_px * 4
        ped_surface = pygame.Surface((canvas_size, canvas_size), pygame.SRCALPHA)
        center_of_canvas = (canvas_size // 2, canvas_size // 2)
        
        # 4. Draw Components
        # Shoulders
        shoulder_rect.center = (center_of_canvas[0] - body_radius_px * 0.7, center_of_canvas[1])
        pygame.draw.ellipse(ped_surface, body_color, shoulder_rect)
        shoulder_rect.center = (center_of_canvas[0] + body_radius_px * 0.7, center_of_canvas[1])
        pygame.draw.ellipse(ped_surface, body_color, shoulder_rect)
        
        # Body (Circle) with White Outline for Contrast
        pygame.draw.circle(ped_surface, body_color, center_of_canvas, body_radius_px)
        pygame.draw.circle(ped_surface, (255, 255, 255), center_of_canvas, body_radius_px, 2) # White Outline

        # Head (Dark Dot)
        head_radius = int(body_radius_px * 0.5)
        head_color = self.config['colors']['pedestrian_head']
        # Offset head slightly in direction of travel (up in local coords)
        head_center = (center_of_canvas[0], center_of_canvas[1] - body_radius_px * 0.4)
        pygame.draw.circle(ped_surface, head_color, head_center, head_radius)

        # 5. Rotate and Blit
        angle_deg = np.rad2deg(-heading_rad)
        rotated_surface = pygame.transform.rotate(ped_surface, angle_deg)
        rotated_surface.set_alpha(alpha)
        new_rect = rotated_surface.get_rect(center=tuple(center_xy_pixels))
        surface.blit(rotated_surface, new_rect.topleft)
        

    def _draw_cyclist_shape(self, surface: pygame.Surface, center_xy_pixels: np.ndarray,
                            length_m: float, heading_rad: float, body_color: Tuple[int, int, int],
                            wheel_color: Tuple[int, int, int], alpha: int = 255):
        """
        --- UPDATED: SYMBOLIC SCALING (The "Giant Triangle") ---
        Enforces a minimum pixel size so Cyclists act as high-visibility markers.
        """
        # 1. Dimensions with a Floor
        # Physical length OR Minimum 22px (huge, impossible to miss)
        l_px = max(22, length_m * self.config['pixels_per_meter']) 
        w_px = l_px * 0.75 # Wide, distinct arrowhead
        
        # 2. Define Vertices (Pointing Right)
        half_l = l_px / 2.0
        half_w = w_px / 2.0
        local_vertices = np.array([
            [half_l, 0],          # Tip
            [-half_l, -half_w],   # Back Left
            [-half_l, half_w]     # Back Right
        ])
        
        # 3. Rotate and Translate
        angle_rad = -heading_rad
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = local_vertices @ rot_matrix.T
        translated_vertices = rotated_vertices + center_xy_pixels
        
        # 4. Draw with Alpha
        min_x, min_y = np.min(translated_vertices, axis=0)
        max_x, max_y = np.max(translated_vertices, axis=0)
        shape_size = (int(max_x - min_x) + 4, int(max_y - min_y) + 4)
        
        if shape_size[0] < 1 or shape_size[1] < 1: return
        shape_surface = pygame.Surface(shape_size, pygame.SRCALPHA)
        local_draw_verts = translated_vertices - [min_x, min_y] + [2, 2] # +2 buffer
        
        # Draw High-Contrast Triangle
        pygame.draw.polygon(shape_surface, body_color, local_draw_verts) # Cyan Fill
        pygame.draw.polygon(shape_surface, (255, 255, 255), local_draw_verts, 3) # THICK White Outline
        
        shape_surface.set_alpha(alpha)
        surface.blit(shape_surface, (int(min_x - 2), int(min_y - 2)))
        
    # --- Low-Level Geometry Helpers (We will reuse/refactor these later) ---
    
    def _world_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        """
        Converts world coordinates to pixel coordinates with EGO-CENTRIC ROTATION.
        """
        if coords.ndim == 1: coords = coords.reshape(1, -1)
        
        # 1. Center the world on the SDC
        centered = coords - self.scene_center_world
        
        # 2. Apply Rotation (New!)
        # x' = x cos - y sin
        # y' = x sin + y cos
        rotated_x = centered[:, 0] * self.cos_rot - centered[:, 1] * self.sin_rot
        rotated_y = centered[:, 0] * self.sin_rot + centered[:, 1] * self.cos_rot
        
        # Stack back into (N, 2) array
        rotated_coords = np.column_stack((rotated_x, rotated_y))

        # 3. Scale and Flip Y (Standard Pygame conversion)
        pixel_coords = rotated_coords * self.config['pixels_per_meter']
        pixel_coords[:, 1] *= -1 # Flip Y because Pygame 0 is top
        
        # 4. Center on Screen
        pixel_coords += self.render_offset_pixels
        return pixel_coords.astype(int)


    def close(self):
        pygame.quit()

# --- Example Usage Block for Testing ---
if __name__ == '__main__':
    from src.utils.config_loader import load_config
    
    print("--- Running ScenarioRendererV6 Test ---")
    config = load_config()
    
    # We'll use the same complex intersection scenario for our test
    scenario_id_to_test = "60b492b6af054262"
    npz_path = os.path.join(config['data']['processed_npz_dir'], 'validation', f"{scenario_id_to_test}.npz")
    
    # Use our V4 parser to get the rich, structured data
    scenario_data = load_npz_scenario(npz_path)
    
    # Initialize the new renderer
    renderer = ScenarioRendererV6(scenario_data, config['renderer_v6'])
    
    # Define the output path
    output_dir = "outputs/v6_renders"
    os.makedirs(output_dir, exist_ok=True)
    output_gif_path = os.path.join(output_dir, f"{scenario_id_to_test}_v6.gif")
    
    # Run the main rendering method
    renderer.render_to_gif(output_gif_path)
    renderer.close()