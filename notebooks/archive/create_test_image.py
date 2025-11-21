import pygame

# --- Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ROAD_WIDTH = 150
BACKGROUND_COLOR = (211, 211, 211) # Light Gray
ROAD_COLOR = (105, 105, 105)       # Dark Gray
CAR_1_COLOR = (255, 0, 0)         # Red
CAR_2_COLOR = (0, 0, 255)         # Blue

# --- Pygame Initialization ---
pygame.init()

# Create the display surface (we won't show it, just draw on it)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Scenario Renderer Test")

# --- Drawing ---
# Fill the background
screen.fill(BACKGROUND_COLOR)

# Draw the vertical road
pygame.draw.rect(screen, ROAD_COLOR, (SCREEN_WIDTH // 2 - ROAD_WIDTH // 2, 0, ROAD_WIDTH, SCREEN_HEIGHT))

# Draw the horizontal road
pygame.draw.rect(screen, ROAD_COLOR, (0, SCREEN_HEIGHT // 2 - ROAD_WIDTH // 2, SCREEN_WIDTH, ROAD_WIDTH))

# --- Draw the two "crashed" cars ---
# We will use pygame.transform.rotate to angle them
# Car 1 (Red)
car_1_surface = pygame.Surface((60, 30)) # Create a surface for the car
car_1_surface.fill(CAR_1_COLOR)
rotated_car_1 = pygame.transform.rotate(car_1_surface, 45) # Rotate it
screen.blit(rotated_car_1, (SCREEN_WIDTH // 2 - 40, SCREEN_HEIGHT // 2 - 50))

# Car 2 (Blue)
car_2_surface = pygame.Surface((60, 30)) # Create a surface for the car
car_2_surface.fill(CAR_2_COLOR)
rotated_car_2 = pygame.transform.rotate(car_2_surface, 120) # Rotate it
screen.blit(rotated_car_2, (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 10))


# --- Save the Image and Quit ---
output_filename = "pygame_test.jpg"
pygame.image.save(screen, output_filename)
pygame.quit()

print(f"Successfully created test image: {output_filename}")