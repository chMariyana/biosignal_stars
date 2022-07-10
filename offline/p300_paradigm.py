#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
"""

import pygame
import random
import json
import os

import pygame as pg
import numpy as np

from pylsl import local_clock, StreamInfo, StreamOutlet

# Load the configurations of the paradigm
with open(os.path.join(os.path.curdir, 'configs', 'config_long_red-yellow.json'), 'r') as file:
    params = json.load(file)

# Define custom colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Initialise Pygame
pg.init()

# Set the window size parameters
sc = pg.display.Info()
WIDTH = sc.current_w
HEIGHT = sc.current_h
WINDOW_WIDTH = WIDTH # 600
WINDOW_HEIGHT = HEIGHT # 600

class Paradigm(object):

    def __init__(self, window, durations, targets):
        self.window = window
        self.durations = durations
        self.targets = targets
        self.tot_trials = None
        self.tot_blocks = None

        # Define the marker stream
        info = StreamInfo(name='p300_markers',
                          type='Markers',
                          channel_count=1,
                          nominal_srate=0,
                          channel_format='string',
                          source_id='p300_markers')
        self.marker_stream = StreamOutlet(info)

        # Define screen parameters
        self.window_width, self.window_height = window.get_size()
        self.center_x = self.window_width / 2
        self.center_y = self.window_height / 2

        # Surface for the arrows to be shown in.
        self.arrow_surf = pg.Surface((self.window_width, self.window_height))
        self.arrow_surf.fill(BLACK)
        self.arrow_surf_rect = self.arrow_surf.get_rect()

        # Define the stimuli.
        self.grid = [
        " ^ ",
        "< >",
        " v ",
        ]
        self.possible_targets = [(0, 1), (1, 0), (1, 2), (2, 1)]
        self.lines = len(self.grid)
        self.columns = len(self.grid[0])
        self.column_width = self.window_width / self.columns
        self.lines_height = self.window_height / self.lines
        self.horizontal_margin = self.column_width / 4
        self.vertical_margin = self.lines_height / 4

        # Define the fonts.
        self.arrow_font = pygame.font.SysFont("None", 190)
        self.msg_font = pg.font.Font(pg.font.get_default_font(), 50)

    def clear_screen(self):
        # Reset the arrow surface.
        self.arrow_surf.fill(BLACK)

        # Clear the screen.
        self.window.fill(BLACK)

        # Update the screen
        pg.display.flip()

        print('Screen cleared')

    def get_letter_surface(self, target_letter, colour=WHITE):
        letter_surface = self.arrow_font.render(target_letter, True, colour)
        #letter_surface = letter_surface.convert_alpha()
        return letter_surface

    def draw_arrows(self, target, highlighted, is_target, display_time=None):
        # TODO: Add cues input which determines which arrows to show.
        # Define the arrows.
        # TODO: Define a better grid.
        # TODO: Implement a resting cue.

        colour = WHITE  

        if highlighted:
            colour = params['highlight_color']
        elif is_target:
            colour = params['target_color']

        # Draw the arrows.
        for y in range(self.lines):
            for x in range(self.columns):
                if target is not None and y == target[0] and x == target[1]:
                    letter_surface = self.get_letter_surface(self.grid[y][x],
                                                            colour=colour)
                else:
                    letter_surface = self.get_letter_surface(self.grid[y][x],
                                                            colour=WHITE)
                self.window.blit(letter_surface,
                        (self.column_width * x + self.horizontal_margin, 
                        self.lines_height * y + self.vertical_margin))

        # Update the screen
        pg.display.flip()

        if display_time is not None:
            pg.time.delay(int(display_time*1000))

            # Draw back to white.
            self.draw_arrows(None, False, False)

    def convert_target(self, target):
        # Converts the target from name to position.

        if target == 'up':
            return self.possible_targets[0]
        if target == 'left':
            return self.possible_targets[1]
        if target == 'right':
            return self.possible_targets[2]
        if target == 'down':
            return self.possible_targets[3]
    
    def convert_position(self, position):
        # Converts the target from position to name.
        if position == self.possible_targets[0]:
            return 'up'
        if position == self.possible_targets[1]:
            return 'left'
        if position == self.possible_targets[2]:
            return 'right'
        if position == self.possible_targets[3]:
            return 'down'

    def draw_baseline_fix(self, delay=1):
        letter_surface = self.get_letter_surface('+', colour=WHITE)
        letter_surface_rect = letter_surface.get_rect()
        letter_surface_rect.center = (self.center_x, self.center_y)
        self.window.blit(letter_surface, letter_surface_rect)

        # Update the screen.
        pg.display.flip()

        if delay is not None:
            pg.time.delay(int(delay*1000))

    def get_targets(self):
        # Create an empty list for the order of targets to run through.
        pick_list = []

        if self.tot_trials is not None:
            if self.tot_trials < len(self.targets):
                raise ValueError(
                    'Number of trials should be at least the same as the number of targets!'
                )
            
            # Get the quotient of the division of desired number of trials
            # and the number of targets.
            tmp = self.tot_trials // len(self.targets)
            # Multiply the targets list with the quotient to get
            # an equal number of instances for each target.
            pick_list = self.targets * tmp

            # Get the remainder of the divsion.
            remainder = self.tot_trials % len(self.targets)

            # If the remainder is not 0
            if remainder != 0:
                # Add random targets to the list until the desired number
                # of trials is reached.
                for _ in range(remainder):
                    pick_list.append(np.random.choice(self.targets))
                
            # Shuffle the target order for randomness
            np.random.shuffle(pick_list)

            return pick_list
    
    def is_adjacency(self, tuple_list):
        # Gets as input a list of tuples.
        # Checks if any two tuples that are directly after each other
        # have exactly the same elements.

        # If there is only one element in the list, there is no repetition.
        if len(tuple_list) == 1:
            return False

        previous_tuple = tuple_list[0]
        for tuple in tuple_list[1:]:
            if previous_tuple == tuple:
                return True
            previous_tuple = tuple
        
        return False
    
    def shuffle(self, highlights):
        # Shuffles the highlights until there is no repeating neighbours.
        # Initial shuffle.
        random.shuffle(highlights) # Changes highlights in place.

        while self.is_adjacency(highlights):
            random.shuffle(highlights)
        
        return highlights

    def run_exp (self, tot_trials, tot_blocks):
        # Set the class attributes.
        self.tot_trials = tot_trials
        self.tot_blocks = tot_blocks

        # Initialise the number of trials and blocks
        trial_num = 1
        block_num = 1

        # Initialise the flags for different stages of the experiment.
        welcome = True
        trail_break = False
        block_break = False

        window_open = True
        run_exp = False
        exp_end = False

        # Define the messages to be shown.
        msg_exp_start = self.msg_font.render(
            'Welcome to the recording. Press any key to continue.',
            False,
            WHITE
        )
        msg_exp_end = self.msg_font.render(
            'Press key UP to start a new block, key DOWN to end the recording.',
            False,
            WHITE
        )

        # Get the rectangle surrounding the text.
        msg_exp_start_rect = msg_exp_start.get_rect()
        msg_exp_end_rect = msg_exp_end.get_rect()

        # Set the center of the rectangular object
        msg_exp_start_rect.center = (self.center_x, self.center_y)
        msg_exp_end_rect.center = (self.center_x, self.center_y)

        # Display the welcome message.
        self.window.blit(msg_exp_start, msg_exp_start_rect)
        # Update the screen.
        pg.display.flip()

        while window_open:
            # Get all the current events.
            for event in pg.event.get():
                # If the window is closed.
                if event.type == pg.QUIT:
                    pg.display.quit()
                    window_open = False # Exit the while loop.

                # If a key is pressed.
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        pg.display.quit()
                        window_open = False # Exit the while loop.
                    else:
                        if welcome:
                            # Empty the screen.
                            paradigm.clear_screen()
                            pg.display.flip()

                            # Get a list of targets to be shown.
                            targets = paradigm.get_targets()
                            print(f'Block {block_num}: {targets}')

                            # Get out of the welcoming stage.
                            welcome = False
                            run_exp = True
                        
                        # If the total amount of blocks has been reached.
                        elif exp_end:
                            if event.key == pg.K_DOWN:
                                pg.display.quit()
                                window_open = False # Exit the while loop.
                            
                            elif event.key == pg.K_UP:
                                # Clear the window.
                                paradigm.clear_screen()

                                # Get a new set of targets.
                                targets = paradigm.get_targets()
                                print(f'Block {block_num}: {targets}')

                                # Reset the flags.
                                block_break = False
                                run_exp = True
                                exp_end = False
            
            if run_exp:
                # Trial start

                # Show the baseline fix.
                # Again, push the sample first because the drawing will hang.
                self.marker_stream.push_sample(
                    [f'baseline_for_trial_{trial_num}-{local_clock()}']
                )
                paradigm.draw_baseline_fix(delay=params['baseline_length'])
                paradigm.clear_screen()

                # Add a delay between the baseline and drawing the arrows.
                pg.time.delay(int(params['delay_baseline_arrows'] * 1000))

                # Show the targets.
                paradigm.draw_arrows(target=None, highlighted=False, is_target=False)

                # Push the marker to the LSL stream.
                self.marker_stream.push_sample(
                    [f'trial_begin_{trial_num}-{local_clock()}']
                )

                # Push the marker to the LSL stream.
                # Pushing the marker before the drawing because I don't know
                # how long the drawing takes to return.
                target = targets[trial_num - 1]
                self.marker_stream.push_sample([f'target_{target}-{local_clock()}'])

                # Display the target.
                target_position = self.convert_target(target)
                paradigm.draw_arrows(target_position,
                                     highlighted=False,
                                     is_target=True,
                                     display_time=params['target_length'])

                # Now do all the highlighting of arrows at random.
                # We want to guarantee that the target is flashed.
                # Thus we take the possible targets as a basis, and add more highlights if necessary.
                times_basis = int(np.floor(params['num_highlights'] / len(self.possible_targets)))
                highlights = self.possible_targets * times_basis
                difference = params['num_highlights'] - (len(self.possible_targets) * times_basis)
                for _ in range(difference):
                    highlights.append(random.choice(self.possible_targets))

                # Shuffle the highlights until there is no adjacency of the same target.
                highlights = self.shuffle(highlights)

                for highlight in highlights:

                    # Push sample before drawing, because it'll wait during the drawing!
                    highlight_string = self.convert_position(highlight)
                    self.marker_stream.push_sample([f'highlight_{highlight_string}-{local_clock()}'])
                    paradigm.draw_arrows(highlight, 
                                        highlighted=True,
                                        is_target=False,
                                        display_time=params['highlight_length'])

                # At the end of the trial, clear the screen again.
                paradigm.clear_screen()

                # If the total number of trials within a block is reached.
                if trial_num == self.tot_trials:
                    # If there are still blocks to run.
                    if block_num < tot_blocks:
                        # Set the trial number to 0, as it will be incremented
                        # at the end of the loop.
                        trial_num = 0
                        # Set the flag to show the block end message.
                        block_break = True
                    # If there are no more blocks to run.
                    else:
                        # Reset the trial number in case another block is chosen to run.
                        trial_num = 0
                        # Set the flag to show the block end message.
                        block_break = True
                # If there are still trials to run in the current block.
                else:
                    # Set the flag to go to the block break.
                    trial_break = True
                    
                # Trial break
                if trial_break:
                    # We don't want to flash the target arrows.
                    #paradigm.clear_screen()

                    # Push the marker to the LSL stream.
                    self.marker_stream.push_sample([f'pause-{local_clock()}'])

                    # Calculate the random pause duration.
                    wait_duration = params['inter_trial_length'] \
                            + np.round(np.random.uniform(0, 1), 2)
                    
                    # Start the timer.
                    timer_start = local_clock()
                    while local_clock() - timer_start < wait_duration:
                        pass

                    # Reset the flag.
                    trial_break = False

                # Block break.
                elif block_break:
                    
                    # Prepare the message.
                    msg_block_end = self.msg_font.render(
                        f'Block #{block_num} has ended.',
                        False,
                        WHITE
                    )
                    msg_block_end_rect = msg_block_end.get_rect()
                    msg_block_end_rect.center = (self.center_x, self.center_y)

                    # Empty the screen.
                    paradigm.clear_screen()

                    # Show the message
                    window.blit(msg_block_end, msg_block_end_rect)

                    # Update the screen.
                    pg.display.flip()

                    # Push the marker to the LSL stream.
                    self.marker_stream.push_sample(
                        [f'block_end_{block_num}-{local_clock()}']
                    )

                    # Start the timer.
                    timer_start = local_clock()
                    while local_clock() - timer_start < params['inter_block_length']:
                        pass

                    # Update the screen.
                    paradigm.clear_screen()

                    # Increment the block number.
                    block_num += 1
                    # Reset the flag.
                    block_break = False

                    # If the total number of blocks is exceeded:
                    if block_num > tot_blocks:
                        # Show the message.
                        window.blit(msg_exp_end, msg_exp_end_rect)

                        # Update the screen.
                        pg.display.flip()

                        # Set the flags.
                        block_break = False
                        run_exp = False
                        exp_end = True
                    else:
                        # Get a new set of targets.
                        targets = paradigm.get_targets()
                        print(f'Block {block_num}: {targets}')
                
                # Increment the trial number.
                trial_num += 1

#currently the scripts are written to be run as standalone
#routines. We should change these to work in conjunction once we get
#the classifiers working sometime in the future.
if __name__=="__main__":

    # Define the targets that are going to be shown
    targets = ['left', 'right', 'up', 'down']



    # Create a Pygame window
    window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pg.display.set_caption('P300 Paradigm')
    window.fill(BLACK)

    # Initialise the Paradigm object to manage the functions and variables.
    paradigm = Paradigm(window, params, targets)

    # Run the experiment.
    paradigm.run_exp(tot_trials=8, tot_blocks=2)