import pygame
from PIL import Image

# --- Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ROAD_WIDTH = 150
BACKGROUND_COLOR = (211, 211, 211)
ROAD_COLOR = (105, 105, 105)
CAR_1_COLOR = (255, 0, 0)
CAR_2_COLOR = (0, 0, 255)

# --- Vehicle Constants (Canonical Representation) ---
CAR_WIDTH = 60
CAR_HEIGHT = 30

# --- Animation Configuration ---
FPS = 10
DURATION_SEC = 3
TOTAL_FRAMES = FPS * DURATION_SEC
COLLISION_FRAME = 18

# --- Vehicle State Configuration ---
CAR_SPEED_PIXELS_PER_FRAME = 20
car_1_start_pos = (SCREEN_WIDTH // 2 - (CAR_HEIGHT // 2), -CAR_WIDTH - 50) # Centered on the vertical road
car_2_start_pos = (-CAR_WIDTH, SCREEN_HEIGHT // 2 - (CAR_HEIGHT // 2)) # Centered on the horizontal road

# --- Pygame Initialization ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Scenario Renderer")

# --- Create Canonical Base Surfaces for Cars (Do this ONLY ONCE) ---
base_car_1_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
base_car_1_surface.fill(CAR_1_COLOR)

base_car_2_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
base_car_2_surface.fill(CAR_2_COLOR)

frames_for_gif = []

# --- Main Animation Loop ---
for frame_num in range(TOTAL_FRAMES):
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, ROAD_COLOR, (SCREEN_WIDTH // 2 - ROAD_WIDTH // 2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, ROAD_COLOR, (0, SCREEN_HEIGHT // 2 - ROAD_WIDTH // 2, SCREEN_WIDTH, ROAD_WIDTH))

    if frame_num < COLLISION_FRAME:
        # --- PRE-COLLISION: Cars are moving ---

        # Car 1 (Red) - Rotated 90 degrees to move vertically
        car_1_y = car_1_start_pos[1] + (frame_num * CAR_SPEED_PIXELS_PER_FRAME)
        rotated_car_1 = pygame.transform.rotate(base_car_1_surface, 90)
        # Get the new bounding box and set its center to our desired position
        car_1_rect = rotated_car_1.get_rect(center=(car_1_start_pos[0] + (CAR_HEIGHT // 2), car_1_y + (CAR_WIDTH // 2)))
        screen.blit(rotated_car_1, car_1_rect.topleft)

        # Car 2 (Blue) - No rotation (0 degrees) to move horizontally
        car_2_x = car_2_start_pos[0] + (frame_num * CAR_SPEED_PIXELS_PER_FRAME)
        # No rotation needed, but we still get the rect for consistent positioning
        car_2_rect = base_car_2_surface.get_rect(center=(car_2_x + (CAR_WIDTH // 2), car_2_start_pos[1] + (CAR_HEIGHT // 2)))
        screen.blit(base_car_2_surface, car_2_rect.topleft)

    else:
        # --- POST-COLLISION: Draw the static crashed state with rotations ---
        
        # Car 1 (Red) - Rotated from its canonical base surface
        rotated_car_1 = pygame.transform.rotate(base_car_1_surface, 45)
        screen.blit(rotated_car_1, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 50))

        # Car 2 (Blue) - Rotated from its canonical base surface
        rotated_car_2 = pygame.transform.rotate(base_car_2_surface, 120)
        screen.blit(rotated_car_2, (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 10))

    # --- Capture the Frame for the GIF ---
    frame_data = pygame.image.tostring(screen, 'RGB')
    pil_image = Image.frombytes('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), frame_data)
    frames_for_gif.append(pil_image)

# --- Save the Collected Frames as a GIF ---
output_filename = "collision_scenario_v2.gif"
frames_for_gif[0].save(
    output_filename,
    save_all=True,
    append_images=frames_for_gif[1:],
    duration=1000 // FPS,
    loop=0
)

pygame.quit()
print(f"Successfully created physically consistent GIF: {output_filename}")