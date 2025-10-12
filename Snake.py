import pygame, random
pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

snake = [[200, 200], [190, 200], [180, 200]]
food = [random.randint(0, 39) * 10, random.randint(0, 39) * 10]
direction = [10, 0]

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: direction = [0, -10]
            if event.key == pygame.K_DOWN: direction = [0, 10]
            if event.key == pygame.K_LEFT: direction = [-10, 0]
            if event.key == pygame.K_RIGHT: direction = [10, 0]
    
    head = [snake[0][0] + direction[0], snake[0][1] + direction[1]]
    snake.insert(0, head)
    
    if head == food:
        food = [random.randint(0, 39) * 10, random.randint(0, 39) * 10]
    else:
        snake.pop()
    
    if head in snake[1:] or not (0 <= head[0] < 400 and 0 <= head[1] < 400):
        break
    
    screen.fill((0, 0, 0))
    for segment in snake:
        pygame.draw.rect(screen, (0, 255, 0), (segment[0], segment[1], 10, 10))
    pygame.draw.rect(screen, (255, 0, 0), (food[0], food[1], 10, 10))
    pygame.display.flip()
    clock.tick(10)
