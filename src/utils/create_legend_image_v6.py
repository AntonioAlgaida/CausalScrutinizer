import os
import sys
import pygame
import numpy as np
from PIL import Image

# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.rendering.scenario_renderer_v6 import ScenarioRendererV6

class LegendRenderer(ScenarioRendererV6):
    def __init__(self, config: dict):
        # No super().__init__ because we lack scenario data
        self.config = config
        self.screen_width = 704
        self.screen_height = 1152 # <-- INCREASED HEIGHT to fit new items
        
        pygame.init()
        try:
            self.font = pygame.font.SysFont("dejavusans", 24, bold=True)
            self.stop_sign_font = pygame.font.SysFont("dejavusans", 10, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)
            self.stop_sign_font = pygame.font.Font(None, 10)

    def generate_legend(self, output_path: str):
        print("--- Generating V6 Visual Legend (Enhanced) ---")
        
        surface = pygame.Surface((self.screen_width, self.screen_height))
        bg_color = self.config['colors']['background_verge']
        surface.fill(bg_color)

        # Title
        title_surf = self.font.render("Visual Legend: Simulation Objects", True, (255, 255, 255))
        surface.blit(title_surf, (30, 16))

        y_cursor = 100
        
        # 1. AV
        self._draw_legend_item(surface, y_cursor, "The AV (You)", self._draw_av_example)
        
        # 2. Other Vehicles
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Other Vehicles", self._draw_vehicle_example)

        # 3. Pedestrians
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Pedestrians (w/ Head Dot)", self._draw_ped_example)

        # 4. Cyclists
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Cyclists", self._draw_cyclist_example)

        # --- NEW ITEM: BIKE LANE ---
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Bike Lane (Dark Red)", self._draw_bike_lane_example)

        # --- NEW ITEM: DRIVEWAY ---
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Driveway (Entrance)", self._draw_driveway_example)

        # --- NEW ITEM: LANE MARKINGS ---
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Centerlines (Gray) &\nLane Markings (W/Y)", self._draw_markings_example)
              
        # 5. Traffic Lights
        y_cursor += 120
        self._draw_legend_item(surface, y_cursor, "Traffic Lights (Bars)", self._draw_tl_example)

        # 6. Stop Signs
        y_cursor += 120
        self._draw_legend_item(surface, y_cursor, "Stop Signs", self._draw_stop_sign_example)

        # 7. Crosswalks
        y_cursor += 100
        self._draw_legend_item(surface, y_cursor, "Crosswalks", self._draw_crosswalk_example)

        # Save
        frame_data = pygame.surfarray.array3d(surface)
        frame_data = np.rot90(frame_data, k=3)
        frame_data = np.fliplr(frame_data)
        
        img = Image.fromarray(frame_data)
        img.save(output_path)
        print(f"✅ V6 Legend saved to: {output_path}")

    def _draw_legend_item(self, surface, y_pos, text, draw_func):
        lines = text.split('\n')
        y_offset = y_pos - 15
        for line in lines:
            text_surf = self.font.render(line, True, (255, 255, 255))
            surface.blit(text_surf, (50, y_offset))
            y_offset += 30
        center_xy = np.array([500, y_pos])
        draw_func(surface, center_xy)

    # --- Drawing Implementations ---

    def _draw_av_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 200, 80)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], rect)
        trail_start = center_xy - np.array([80, 0])
        self._draw_dashed_line(surface, trail_start, center_xy, self.config['colors']['av'], 2, 7, 7)
        self._draw_sumo_vehicle_shape(surface, center_xy, 2.0, 4.8, 0, self.config['colors']['av'])

    def _draw_vehicle_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 200, 80)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], rect)
        self._draw_sumo_vehicle_shape(surface, center_xy, 2.0, 4.8, 0, self.config['colors']['vehicle'])

    def _draw_ped_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 100, 80)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_sidewalk'], rect)
        self._draw_sumo_pedestrian(surface, center_xy, 0.8, np.pi/2, self.config['colors']['pedestrian_body'])

    def _draw_cyclist_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 150, 80)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], rect)

        # Draw the new High-Vis Triangle
        self._draw_cyclist_shape(
            surface, center_xy, 
            length_m=2.0, heading_rad=0, # Facing RIGHT
            body_color=self.config['colors']['cyclist_body'],
            wheel_color=self.config['colors']['bicycle_wheels']
        )
    # --- NEW: Bike Lane Example ---
    def _draw_bike_lane_example(self, surface, center_xy):
        road_rect = pygame.Rect(0, 0, 200, 80)
        road_rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], road_rect)
        
        # Draw Bike Lane
        lane_height = 30
        lane_rect = pygame.Rect(road_rect.left, center_xy[1] - lane_height//2, road_rect.width, lane_height)
        pygame.draw.rect(surface, self.config['colors']['surface_bike_lane'], lane_rect)
        
        # Draw Cyclist inside
        self._draw_cyclist_shape(
            surface, center_xy, 
            length_m=1.8, heading_rad=0, 
            body_color=self.config['colors']['cyclist_body'], 
            wheel_color=self.config['colors']['bicycle_wheels']
        )
    # --- NEW: Driveway Example ---
    def _draw_driveway_example(self, surface, center_xy):
        # Draw main road at bottom
        road_rect = pygame.Rect(0, 0, 200, 40)
        road_rect.center = center_xy + np.array([0, 30])
        pygame.draw.rect(surface, self.config['colors']['surface_road'], road_rect)
        
        # Draw Driveway connecting from top
        driveway_rect = pygame.Rect(0, 0, 60, 60)
        driveway_rect.midbottom = road_rect.midtop
        pygame.draw.rect(surface, self.config['colors']['surface_driveway'], driveway_rect)

    def _draw_tl_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 300, 100)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], rect)
        line_y1, line_y2 = center_xy[1] - 20, center_xy[1] + 20
        pygame.draw.line(surface, (255,255,255), (center_xy[0]-100, line_y1), (center_xy[0]+100, line_y1), 2)
        pygame.draw.line(surface, (255,255,255), (center_xy[0]-100, line_y2), (center_xy[0]+100, line_y2), 2)
        
        bar_w, bar_h = 10, 30
        r_center = center_xy - np.array([50, 0])
        pygame.draw.rect(surface, self.config['colors']['tl_stop'][:3], (r_center[0], r_center[1]-15, bar_w, bar_h))
        y_center = center_xy
        pygame.draw.rect(surface, self.config['colors']['tl_caution'][:3], (y_center[0], y_center[1]-15, bar_w, bar_h))
        g_center = center_xy + np.array([50, 0])
        pygame.draw.rect(surface, self.config['colors']['tl_go'][:3], (g_center[0], g_center[1]-15, bar_w, bar_h))

    def _draw_stop_sign_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 100, 100)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_sidewalk'], rect)
        radius_m = self.config['radii']['stop_sign_m']
        self._draw_regular_polygon(surface, center_xy, radius_m, 5, self.config['colors']['stop_sign'])
        text_surf = self.stop_sign_font.render("STOP", True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=tuple(center_xy))
        surface.blit(text_surf, text_rect)

    def _draw_crosswalk_example(self, surface, center_xy):
        rect = pygame.Rect(0, 0, 200, 80)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], rect)
        white = self.config['colors']['marking_white']
        w, h = 80, 40
        for i in range(int(center_xy[0] - w/2), int(center_xy[0] + w/2), 12):
            pygame.draw.line(surface, white, (i, center_xy[1]-h/2), (i, center_xy[1]+h/2), 6)

    def _draw_markings_example(self, surface, center_xy):
        # Draw a road surface
        rect = pygame.Rect(0, 0, 250, 80)
        rect.center = center_xy
        pygame.draw.rect(surface, self.config['colors']['surface_road'], rect)
        
        y_pos = center_xy[1]
        x_start, x_end = rect.left + 20, rect.right - 20
        
        # 1. Double Yellow (Left side - Divider)
        yellow = self.config['colors']['marking_yellow']
        pygame.draw.line(surface, yellow, (x_start, y_pos + 15), (x_end, y_pos + 15), 2)
        pygame.draw.line(surface, yellow, (x_start, y_pos + 19), (x_end, y_pos + 19), 2)
        
        # 2. Dashed White (Right side - Lane Line)
        white = self.config['colors']['marking_white']
        dashed_y = y_pos - 15
        self._draw_dashed_line(surface, (x_start, dashed_y), (x_end, dashed_y), white, 2, 7, 7)

        # 3. NEW: Centerline (Gray) - Running through the "middle" of the top lane
        # This represents the logical graph connection.
        gray = self.config['colors']['marking_centerline']
        centerline_y = y_pos - 30 # In the middle of the top lane
        pygame.draw.line(surface, gray, (x_start, centerline_y), (x_end, centerline_y), 1)
        
if __name__ == '__main__':
    config = load_config()
    legend_gen = LegendRenderer(config['renderer_v6'])
    output_dir = os.path.join(PROJECT_ROOT, "outputs/legend_assets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "visual_legend.png")
    legend_gen.generate_legend(output_path)