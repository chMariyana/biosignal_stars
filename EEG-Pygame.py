
import pygame
from pylsl import StreamInlet, resolve_stream 

#Resolving from the stream 
streams = resolve_stream('type', 'Markers')

# Create a new stream inlet 
inlet = StreamInlet(streams[0])

# Initialization
pygame.init()
size=width,height=25*25+300,25*25+100
screen=pygame.display.set_mode(size) # set window size
width1, height1 = 25, 25  # size of the block that represents the robot
speed = 75  #the speed of movement
x, y = 225, 600  # initial position of the robot
x_p, y_p = 225, 600 # previous position of the robot
last_step = pygame.K_f # record the last step
num_move = 1 # number of movements
list_left  = [pygame.K_a]
list_up    = [pygame.K_w]
list_right = [pygame.K_d]
list_rep   = [pygame.K_s]
list_none  = [pygame.K_z]
WHITE=pygame.Color('white')
BLUE=pygame.Color('blue')
GREEN=pygame.Color('green')
BLACK=pygame.Color('black')
Red = pygame.Color('red')
# draw the grid
def draw_gird():
    for i in range(0,630,25):
        pygame.draw.line(screen, WHITE, (0, i), (630, i), 1)
        pygame.draw.line(screen, WHITE, (i, 0), (i, 630), 1)

# function used to show text on the screen
def message_display(text,text_position_x,text_position_y,textsize):
    largeText = pygame.font.SysFont('arial', textsize)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.bottomleft = (text_position_x,text_position_y)
    screen.blit(TextSurf, TextRect)
    pygame.display.update()


# draw the initial position of the robot
def draw_initial():
    pygame.draw.rect(screen, WHITE, (225, 600, width1, height1))
    for indy in [*range(0,10), *range(12,25)]:
        pygame.draw.rect(screen, GREEN, (7*25, 25*(24-indy), width1, height1))
    
    for indxy in range(0,14):
        pygame.draw.rect(screen, GREEN, ((7+indxy)*25, 25*0, width1, height1))
        pygame.draw.rect(screen, GREEN, (20*25, 25*(7+indxy), width1, height1))

    for indx in range(0,9):
        pygame.draw.rect(screen, GREEN, ((12+indx)*25, 25*5, width1, height1))
        pygame.draw.rect(screen, GREEN, ((12+indx)*25, 25*7, width1, height1))
        pygame.draw.rect(screen, GREEN, ((12+indx)*25, 25*20, width1, height1))
        pygame.draw.rect(screen, GREEN, ((7+indx)*25, 25*12, width1, height1))
        pygame.draw.rect(screen, GREEN, ((7+indx)*25, 25*15, width1, height1))
    
    for indy in range(20,25):
        pygame.draw.rect(screen, GREEN, (12*25, 25*(indy), width1, height1))

    pygame.draw.rect(screen, GREEN, (12*25, 25*6, width1, height1))
    pygame.draw.rect(screen, GREEN, (15*25, 25*13, width1, height1))
    pygame.draw.rect(screen, GREEN, (15*25, 25*14, width1, height1))

    for indy in range(0,4):
        pygame.draw.rect(screen, Red, (20*25, 25*(indy+1), width1, height1))

    # screen.fill(BLACK)
    draw_gird()
    pygame.display.update()

# draw the instruction of the input
# def draw_instrucions():
#     message_display('UP: W', 650, width1*2,25)
#     message_display('LEFT: A ', 650, width1*3,25)
#     message_display('RIGHT: D', 650, width1*4,25)
#     message_display('PRE.MOVE: S ', 650, width1*5,25)
#     message_display('NOTHING: Z', 650, width1*6,25)
#     message_display('EXIT: BACKSPACE ', 650, width1*7, 25)
#     pygame.display.update()

def  leftwall(x,y):
    if (0*25 < y < 12*25) | (15*25 < y < 25*25):
        if (x - speed) > (7*25):
            finalspeed = speed
        elif (x - speed*2/3) > (7*25):
            finalspeed = speed*2/3
        elif (x - speed/3) > (7*25):
            finalspeed = speed/3
        else: 
            finalspeed = 0

    elif (11*25 < y < 16*25):
        if (x - speed) > (15*25):
            finalspeed = speed
        elif (x - speed*2/3) > (15*25):
            finalspeed = speed*2/3
        elif (x - speed/3) > (15*25):
            finalspeed = speed/3
        else: 
            finalspeed = 0
    return finalspeed

def rightwall(x,y):
    if (0*25 < y < 5*25):
        if (x + speed) > (19*25):
            finalspeed = speed
            won = 1
            message_display('You Won', 650, width1*10, 25)
            message_display('PRESS SPACE TO RESTART', 650, width1*12, 25)
        elif (x + speed) < (20*25):
            finalspeed = speed
            won = 0

    elif (4*25 < y < 8*25) | (19*25 < y < 25*25):
        won = 0
        if (x + speed) < (12*25):
            finalspeed = speed
        elif (x + speed*2/3) < (12*25):
            finalspeed = speed*2/3
        elif (x + speed/3) < (12*25):
            finalspeed = speed/3
        else: 
            finalspeed = 0
    
    elif (7*25 < y < 21*25):
        won = 0
        if (x + speed) < (20*25):
            finalspeed = speed
        elif (x + speed*2/3) < (20*25):
            finalspeed = speed*2/3
        elif (x + speed/3) < (20*25):
            finalspeed = speed/3
        else: 
            finalspeed = 0

    return finalspeed,won

def upwall(x,y):
    if (7*25 < x < 16*25) & (15*25 < y < 25*25):
        if (15*25 < (y - speed)):
            finalspeed = speed
        elif (15*25 < (y - speed*2/3)):
            finalspeed = speed*2/3
        elif (15*25 < (y - speed/3)):
            finalspeed = speed/3
        else: 
            finalspeed = 0
    
    elif (15*25 < x < 20*25) & (15*25 < y < 25*25):
        finalspeed = speed

    elif (11*25 < x < 20*25) & (7*25 < y < 16*25):
        if (7*25 < (y - speed)):
            finalspeed = speed
        elif (7*25 < (y - speed*2/3)):
            finalspeed = speed*2/3
        elif (7*25 < (y - speed/3)):
            finalspeed = speed/3
        else: 
            finalspeed = 0

    elif (7*25 < x < 20*25) & (0*25 < y < 9*25):
        if (0*25 < (y - speed)):
            finalspeed = speed
        elif (0*25 < (y - speed*2/3)):
            finalspeed = speed*2/3
        elif (0*25 < (y - speed/3)):
            finalspeed = speed/3
        else: 
            finalspeed = 0
    
    elif (7*25 < x < 12*25) & (7*25 < y < 12*25):
            finalspeed = speed
    
    return finalspeed


# show the robot goes up
def draw_up(x,y,x_p,y_p):

    pygame.draw.rect(screen, BLUE, (x_p, y_p, width1, height1))
    pygame.draw.rect(screen, WHITE, (x, y, width1, height1))
    pygame.draw.line(screen, GREEN, (x_p+12.5, y_p+12.5), (x+12.5, y+25), 2)
    pygame.draw.line(screen, GREEN, (x+7.5, y+30), (x+12.5, y+25), 2)
    pygame.draw.line(screen, GREEN, (x+17.5, y+30), (x+12.5, y+25), 2)

# show the robot goes left
def draw_left(x,y,x_p,y_p):
    pygame.draw.rect(screen, BLUE, (x_p, y_p, width1, height1))
    pygame.draw.rect(screen, WHITE, (x, y, width1, height1))
    pygame.draw.line(screen, GREEN, (x_p+12.5, y_p+12.5), (x+25, y+12.5), 2)
    pygame.draw.line(screen, GREEN, (x+30, y_p+7.5), (x+25, y+12.5), 2)
    pygame.draw.line(screen, GREEN, (x+30, y_p+17.5), (x+25, y+12.5), 2)

# show the robot goes right
def draw_right(x,y,x_p,y_p):
    pygame.draw.rect(screen, BLUE, (x_p, y_p, width1, height1))
    pygame.draw.rect(screen, WHITE, (x, y, width1, height1))
    pygame.draw.line(screen, GREEN, (x_p+12.5, y_p+12.5), (x, y+12.5), 2)
    pygame.draw.line(screen, GREEN, (x-5, y_p+7.5), (x, y+12.5), 2)
    pygame.draw.line(screen, GREEN, (x-5, y_p+17.5), (x, y+12.5), 2)

# text set
def text_objects(text, font):
    textSurface = font.render(text, True, WHITE)
    return textSurface, textSurface.get_rect()

# show the input characters
# def draw_input():
#     if event.key <= 57:
#         message_display(chr(event.key), (num_move-1) * width1, 700, 20)
#     else:
#         message_display(chr(event.key - 32), (num_move-1)*width1, 700, 20)

# main
samples = []
clock = pygame.time.Clock()
while True:
    draw_initial()
    #draw_instrucions()
    sample, timestamp = inlet.pull_sample()
    for event in pygame.event.get():
        if num_move > 40: # terminate it when 4 operations are completed
            if num_move == 41 :
                message_display('You Lost', 650, width1*12, 25)
                message_display('PRESS SPACE TO RESTART', 650, width1*15, 25)
                num_move += 1 # count the number of movements
            if event.type == pygame.KEYUP: # after one input from the keyboard
                if event.key == pygame.K_SPACE: # if input is space, reset the state
                    num_move = 1
                    x, y = 225, 600  # initial position of the robot
                    x_p, y_p = 225, 600 # previous position of the robot
                    screen.fill(BLACK)
                    draw_initial()
        # else:
    if type(sample[0]) is str:
        #draw_input()
        num_move += 1 # record the number of operations
        print(num_move)
        if sample[0] == 'Left':
            finalspeed = leftwall(x,y)
            if finalspeed > 0:
                x_p, y_p = x, y # record the previous position of the robot
                x -= finalspeed
                draw_left(x, y, x_p, y_p)
        if sample[0] == 'Right':
            finalspeed,won = rightwall(x,y)
            if finalspeed > 0:
                x_p, y_p = x, y # record the previous position of the robot
                x += finalspeed
                draw_right(x, y, x_p, y_p)
            if won > 0:                     
                num_move = 50
                
        if sample[0] == 'Up':
            finalspeed = upwall(x,y)
            if finalspeed > 0:
                x_p, y_p = x, y # record the previous position of the robot
                y -= finalspeed
                draw_up(x,y,x_p,y_p)
        draw_gird()
        if x<0:
            x=0
        if x>600:
            x=600
        if y<0:
            y=0
        if y>600:
            y=600
        clock.tick(30)
        pygame.display.update()# update the window
