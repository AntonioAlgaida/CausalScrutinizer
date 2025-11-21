import pygame
import numpy as np
from PIL import Image
import os
from typing import Tuple

# This script is now fully self-contained and does not import from our src/.

# --- Configuration (Copied from ScenarioRenderer for consistency) ---
CONFIG = {
    'pixels_per_meter': 5,
    'bg_color': (30, 30, 30),
    'av_color': (255, 0, 255),
    'vehicle_color': (0, 150, 255),
    'pedestrian_color': (255, 100, 0),
    'cyclist_color': (0, 255, 255),
    'stop_sign_color': (255, 0, 0),
    'tl_go_color': (0, 255, 0),
    'av_trail_color': (255, 0, 255),
    'av_trail_width': 2,
    'av_trail_dash_length': 7,
    'av_trail_gap_length': 7,
    'pedestrian_radius_m': 0.7,
    'traffic_light_radius_m': 1.5,
    'stop_sign_radius_m': 1.5,
    'tl_stop_color': (255, 0, 0, 150),       # Semi-transparent Red
    'tl_caution_color': (255, 255, 0, 150),  # Semi-transparent Yellow
    'tl_go_color': (0, 255, 0, 150),        # Semi-transparent Green
}

# --- Self-Contained Drawing Functions ---

def draw_vehicle_shape(surface: pygame.Surface, center_xy, width_m, length_m, angle_rad, color):
    w_px = width_m * CONFIG['pixels_per_meter']
    l_px = length_m * CONFIG['pixels_per_meter']
    half_l, half_w = l_px / 2.0, w_px / 2.0
    front_taper = 0.7
    
    local_vertices = np.array([
        [-half_l, -half_w], [half_l, -half_w * front_taper],
        [half_l, half_w * front_taper], [-half_l, half_w]
    ])
    
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    rotated_vertices = local_vertices @ rot_matrix.T
    translated_vertices = rotated_vertices + center_xy
    
    pygame.draw.polygon(surface, color, translated_vertices)

def draw_circle(surface: pygame.Surface, center_xy, radius_m, color):
    radius_px = int(radius_m * CONFIG['pixels_per_meter'])
    pygame.draw.circle(surface, color, center_xy, radius_px)
    

def draw_regular_polygon(surface: pygame.Surface, center_xy, radius_m, num_sides, color):
    radius_px = int(radius_m * CONFIG['pixels_per_meter'])
    if radius_px < 1: return
    angle_step = 2 * np.pi / num_sides
    vertices = []
    for i in range(num_sides):
        angle = i * angle_step - (np.pi / 2)
        x = center_xy[0] + radius_px * np.cos(angle)
        y = center_xy[1] + radius_px * np.sin(angle)
        vertices.append((x, y))
    pygame.draw.polygon(surface, color, vertices)

def draw_dashed_line(surface: pygame.Surface, start_pos, end_pos, color, width, dash_len, gap_len):
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
        
        pygame.draw.line(surface, color, (int(current_pos[0]), int(current_pos[1])), (int(dash_end[0]), int(dash_end[1])), width)
        current_pos += unit_dir * (dash_len + gap_len)
        dist_covered += dash_len + gap_len

def main():
    print("--- Generating Visual Legend Image (V3 - Self-Contained) ---")
    
    output_dir = "outputs/legend_assets"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "visual_legend.png")

    pygame.init()
    
    canvas_width, canvas_height = 896, 800
    surface = pygame.Surface((canvas_width, canvas_height))
    surface.fill(CONFIG['bg_color'])
    
    try:
        font = pygame.font.SysFont("dejavusans", 30, bold=True)
    except pygame.error:
        font = pygame.font.Font(None, 36)

    # --- Direct Drawing Logic ---
    vehicle_l, vehicle_w = 4.7, 1.8
    cyclist_l, cyclist_w = 1.8, 0.7
    
    items = {
        "The AV": (80, 'vehicle', CONFIG['av_color'], (vehicle_w, vehicle_l)),
        "Other Vehicle": (160, 'vehicle', CONFIG['vehicle_color'], (vehicle_w, vehicle_l)),
        "Pedestrian": (240, 'circle', CONFIG['pedestrian_color'], CONFIG['pedestrian_radius_m']),
        "Cyclist": (320, 'vehicle', CONFIG['cyclist_color'], (cyclist_w, cyclist_l)),
        "Stop Sign": (400, 'octagon', CONFIG['stop_sign_color'], CONFIG['stop_sign_radius_m']), # <-- CHANGE 'circle' to 'octagon'
        "Traffic Light (Example: Go)": (480, 'circle', CONFIG['tl_go_color'], CONFIG['traffic_light_radius_m']),
        "Traffic Light (Example: Caution)": (560, 'circle', CONFIG['tl_caution_color'], CONFIG['traffic_light_radius_m']),
        "Traffic Light (Example: Stop)": (640, 'circle', CONFIG['tl_stop_color'], CONFIG['traffic_light_radius_m'])
    }

    for text, (y_pos, shape_type, color, dims) in items.items():
        shape_center = (100, y_pos)
        if shape_type == 'vehicle':
            draw_vehicle_shape(surface, shape_center, dims[0], dims[1], np.deg2rad(30), color)
        elif shape_type == 'circle':
            draw_circle(surface, shape_center, dims, color)
        elif shape_type == 'octagon':
            draw_regular_polygon(surface, shape_center, dims, 5, color)
        
        text_render = font.render(text, True, (255, 255, 255))
        surface.blit(text_render, (180, y_pos - text_render.get_height() // 2))
    
    # Draw the trail separately
    y_pos = 720
    draw_dashed_line(surface, (50, y_pos), (150, y_pos), CONFIG['av_trail_color'], CONFIG['av_trail_width'], CONFIG['av_trail_dash_length'], CONFIG['av_trail_gap_length'])
    text_render = font.render("AV's Recent Path", True, (255, 255, 255))
    surface.blit(text_render, (180, y_pos - text_render.get_height() // 2))

    # --- Save the final image ---
    frame_data = pygame.surfarray.array3d(surface)
    frame_data = np.rot90(frame_data, k=3)
    frame_data = np.fliplr(frame_data)
    Image.fromarray(frame_data).save(output_path)
    
    pygame.quit()
    print(f"\n✅ Visual Legend image saved successfully to: {output_path}")

if __name__ == '__main__':
    main()