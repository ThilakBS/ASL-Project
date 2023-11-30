import pygame
import cv2
import torch
from PIL import Image
from torchvision import transforms
from CNN_CLASS import Net
import numpy as np
import copy

spaces = {
 'A': 33,
 'B': 32,
 'C': 34,
 'D': 33,
 'E': 30,
 'F': 29,
 'G': 34,
 'H': 33,
 'I': 14,
 'J': 28,
 'K': 33,
 'L': 31,
 'M': 38,
 'N': 34,
 'O': 34,
 'P': 29,
 'Q': 35,
 'R': 34,
 'S': 30,
 'T': 30,
 'U': 33,
 'V': 32,
 'W': 39,
 'X': 32,
 'Y': 32,
 'Z': 32
}

BACKGROUND_COLOR = (0,0,0)
PREDICTION_BOX_COLOR = (32, 30, 32)
WRITING_BOX_COLOR = (32, 30, 32)
TEXT_COLOR = (126,229,247)
BORDER_COLOR = (255,255,255)

MARGIN = 20
BLOCK_TITLE_FONT_SIZE = 32
PREDICTION_FONT_SIZE = 100
LETTER_FONT_SIZE = 74
BORDER_THICKNESS = 4


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
PATH = 'sign_lang_20.pth'

net = Net()
#net.load_state_dict(torch.load(PATH))
net.eval()

pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill(BACKGROUND_COLOR)

webcam_screen = pygame.Surface((370, 270))
prediction_screen = pygame.Surface((370, 270))
writing_page = pygame.Surface((760, 270))

writing_page.fill((32, 30, 32))

cap = cv2.VideoCapture(0)

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5190, 0.4977, 0.5134), std=(0.2028, 0.2328, 0.2416))
])

block_title_font = pygame.font.Font(None, BLOCK_TITLE_FONT_SIZE)
prediction_font = pygame.font.Font(None, PREDICTION_FONT_SIZE)
letter_font = pygame.font.Font(None, LETTER_FONT_SIZE)

def new_line():
    global index, current_row, current_col

    current_row += 1.25
    current_col = 5

    pygame.display.update()
    pygame.display.flip()

letters_positions = []

def display_text():
    global current_row, current_col, letters_positions

    ret, frame_print = cap.read()
    frame_print = Image.fromarray(cv2.cvtColor(frame_print, cv2.COLOR_BGR2RGB))
    image = transform(frame_print)
    image = image.unsqueeze(0)
    output = net(image)
    _, predicted_class = torch.max(output, 1)
    letter = classes[predicted_class.item()]
    text = letter_font.render(letter, 1, TEXT_COLOR)
    text_pos = text.get_rect(topleft=(current_col, 20 + current_row * 35))
    writing_page.blit(text, text_pos)
    letters_positions.append((text_pos, letter, current_col))
    current_col += spaces[letter]

    if current_col >= writing_page.get_width() - spaces[letter]:
        new_line()

    pygame.display.update()
    pygame.display.flip()

def space():
    global current_row, current_col, letters_positions

    current_col += spaces[' ']

    if current_col >= writing_page.get_width() - spaces[' ']:
        new_line()

    pygame.display.update()
    pygame.display.flip()

def delete_last_letter():
    global letters_positions, current_col

    if letters_positions:
        last_text_pos, last_letter, last_col = letters_positions.pop()
        letter_text = letter_font.render(last_letter, 1, TEXT_COLOR)
        letter_size = letter_text.get_size()
        pygame.draw.rect(writing_page, WRITING_BOX_COLOR, pygame.Rect(last_text_pos.x, last_text_pos.y, *letter_size))
        current_col = last_col
        for text_pos, letter, _ in letters_positions:
            text = letter_font.render(letter, 1, TEXT_COLOR)
            writing_page.blit(text, text_pos)
        pygame.display.update()
        pygame.display.flip()

current_col = 5
current_row = 0

SPACING = 15

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_k:
                display_text()
            if event.key == pygame.K_SPACE:
                space()
            if event.key == pygame.K_TAB:
                new_line()
            if event.key == pygame.K_BACKSPACE:
                print("Backspace pressed")
                delete_last_letter()

    ret, frame = cap.read()
    pred_frame = copy.deepcopy(frame)

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (370, 270))
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = pygame.surfarray.make_surface(frame)
        webcam_screen.blit(frame, (0, 0))

    pred_frame = Image.fromarray(cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB))
    image = transform(pred_frame)
    image = image.unsqueeze(0)
    output = net(image)
    _, predicted_class = torch.max(output, 1)
    prediction_screen.fill(PREDICTION_BOX_COLOR)

    text = prediction_font.render(classes[predicted_class.item()], 1, TEXT_COLOR)
    prediction_screen.blit(text, (182 - text.get_width() // 2, 140 - text.get_height() // 2))

    screen.fill(BACKGROUND_COLOR)

    webcam_title = block_title_font.render("WEBCAM FEED", 1, TEXT_COLOR)
    screen.blit(webcam_title, (20, 10))
    prediction_title = block_title_font.render("PREDICTION", 1, TEXT_COLOR)
    screen.blit(prediction_title, (410, 10))
    writing_title = block_title_font.render("WRITING PAD", 1, TEXT_COLOR)
    screen.blit(writing_title, (20, 290))

    pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(19, 44, 372, 272), BORDER_THICKNESS)
    pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(409, 44, 372, 272), BORDER_THICKNESS)
    pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(19, 324, 762, 272), BORDER_THICKNESS)

    screen.blit(webcam_screen, (20, 45))
    screen.blit(prediction_screen, (410, 45))
    screen.blit(writing_page, (20, 325))
    pygame.display.update()
    pygame.display.flip()



cap.release()
pygame.quit()
