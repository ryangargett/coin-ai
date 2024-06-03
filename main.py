import kivy.app
import kivy.uix.screenmanager
import kivy.uix.image
import random
import kivy.core.audio
import os
import functools
import kivy.uix.behaviors
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
import pickle
import threading
import kivy.base
import random
import time

from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation

import numpy as np

INT_MAX = 2147483647


class DiversityException(Exception):
    pass


def get_distance(pos1, pos2):
    """helper function to calculate Euclidean distance between two positions

    Args:
        pos1 (list): x and y coordinates of the first position
        pos2 (list): x and y coordinates of the second position

    Returns:
        distance (float): Euclidean distance between the two positions
    """

    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return distance


class gIDGenerator(Mutation):

    """Based on Generalized immigration schemes for dynamic evolutionary
    multiobjective optimization: 10.1109/CEC.2011.5949865

    Injects random instances into the population to maintain diversity,
    determined by parameter theta.

    Args:
        alpha (float): percentage of population to be replaced by random
        instances (immigration rate) [0, 1]

    Returns:
        population (Population): population with random instances injected
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def _do(self, problem, X, **kwargs):

        num_immigrants = int(self.alpha * len(X))

        fitness_values = problem.evaluate(X)
        sorted_indices = np.argsort(fitness_values)[::-1]

        for ii in range(num_immigrants):
            worst_index = sorted_indices[ii]
            X[worst_index] = np.random.uniform(problem.xl, problem.xu, problem.n_var)

        return X


class CoinProblem(Problem):
    def __init__(self, coin_pos, enemy_pos):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0, 0]), xu=np.array([1, 1]))

        self.coin_pos = coin_pos
        self.enemy_pos = enemy_pos

    def _evaluate(self, x, out):

        coin_distances = np.array([np.min([get_distance(pos, x[i]) for pos in self.coin_pos])
                                   for i in range(x.shape[0])])

        # Calculate the penalty for each enemy using an inverse cube root law
        enemy_penalties = np.array([np.sum([1 / get_distance(pos, x[i])**(1/3) for pos in self.enemy_pos])
                                    for i in range(x.shape[0])])

        out["F"] = np.column_stack([coin_distances, -enemy_penalties])


class CollectCoinThread(threading.Thread):

    def __init__(self, screen):
        super().__init__()
        self.screen = screen
        self.schema = "multi"
        self.num_population = 3
        self.population = None
        self.sub_pop_size = 50
        self.migration_rate = 0.1
        self.solution_depth = get_termination("n_gen", 20)

        self.get_population_schema()

    def save_population(self, fname):

        if self.num_population > 1:
            for pop, ii in zip(self.population, range(self.num_population)):
                fname = f"{fname}_{ii}"
                with open(fname, 'wb') as f:
                    pickle.dump(pop, f)

        else:
            with open(fname, 'wb') as f:
                pickle.dump(self.population, f)

    def migrate(self, populations):
        """helper function to migrate individuals between populations in a
        multi-population schema with a ring topology. Populations are mutated in-place.

        Args:
            populations (list): list of populations to migrate individuals between
        """

        population_size = len(populations[0].X)
        num_migrants = int(self.migration_rate * population_size)

        for ii, population in enumerate(populations):

            migrants_idx = np.random.choice(population.X.shape[0], num_migrants, replace=False)
            print(f"migrating the following individuals from population {ii}: {migrants_idx}")
            migrants = population.X[migrants_idx]

            sel_population = (ii + 1) % self.num_population
            to_replace = np.random.choice(population_size, num_migrants, replace=False)
            print(f"replacing the following individuals in population {sel_population}: {to_replace}")

            populations[sel_population].X[to_replace] = migrants

    def get_population_schema(self):
        """helper function to get the population based on the requested diversity
        schema, if an invalid schema is returned, a default NSGA-II population
        with no schema is constructed"""

        try:
            if self.schema == "maintain":

                self.population = NSGA2(
                    pop_size=100,
                    eliminate_duplicates=True,
                    mutation=gIDGenerator()
                )

            elif self.schema == "increase":

                self.population = DNSGA2(
                    perc_detect_change=0.1,
                    pop_size=100,
                    eliminate_duplicates=True
                )

            elif self.schema == "multi":

                if self.num_population < 2:
                    raise DiversityException("Number of populations must be greater than 1 for multi-diversity schema")

                self.population = [NSGA2(
                    pop_size=self.sub_pop_size,
                    eliminate_duplicates=True
                ) for pop in range(self.num_population)]

                print(f"population: {self.population}")

            else:
                raise DiversityException(f"Requested diversity schema: {self.schema} not supported")

        except DiversityException as e:
            print(e)

        if self.population is None:
            self.population = NSGA2(
                pop_size=100,
                eliminate_duplicates=True
            )

    def get_new_position(self, coin_pos, enemy_pos):
        """Determines the best position of the player agent to move towards
        based on the solution with best fitness values provided by the genetic
        population(s), also checks to see whether a multi-population schema is
        being used, in which case the population with best solution is selected,
        before performing genetic migration

        Returns:
            coin_pos (list): a list of x and y coordinates for all coins still
            present in the gamespace
            enemy_pos (list): a list of x and y coordinates for all hazards
            (fires and enemies) in the gamespace
        """

        problem = CoinProblem(coin_pos, enemy_pos)

        if self.num_population == 1:

            res = minimize(
                problem,
                self.population,
                termination=self.solution_depth,
                seed=1,
                verbose=True
            )

        else:

            global_results = []
            for population in self.population:

                res = minimize(
                    problem,
                    population,
                    termination=self.solution_depth,
                    seed=1,
                    verbose=True
                )

                global_results.append(res)

            self.migrate(global_results)
            res = min(global_results, key=lambda x: np.min(x.F[:, 0] - x.F[:, 1]))

        best_idx = np.argmin(res.F[:, 0] - res.F[:, 1])
        best_pos = res.X[best_idx].tolist()

        print(f"Calculated best position: {best_pos}")
        return best_pos

    def run(self):

        start_pos = [0.1, 0.1]
        app.start_char_animation(lvl_num, start_pos)
        curr_screen = app.root.screens[lvl_num]
        coin_pos, monster_pos, fire_pos = position_func_v3()
        enemy_pos = monster_pos + fire_pos
        num_coins = len(coin_pos)

        if self.population is not None:
            while num_coins > 0 and curr_screen.character_killed is False:
                best_pos = self.get_new_position(coin_pos, enemy_pos)
                app.start_char_animation(lvl_num, best_pos)
                coin_pos, monster_pos, fire_pos = position_func_v3()
                enemy_pos = monster_pos + fire_pos
                num_coins = len(coin_pos)


def position_func_v3():  # added by Kit 03 July 2023
    curr_screen = app.root.screens[lvl_num]

    coins = curr_screen.coins_ids
    if len(coins.items()) == 0:  # "len(coins.items())" is the number of coins in the current state
        return [], [], []

    coin_pos = []
    for k in range(len(coins.items())):
        try:
            curr_coin = coins[list(coins.keys())[k]]
            curr_coin_center = [
                curr_coin.pos_hint['x'], curr_coin.pos_hint['y']]
            coin_pos.append(curr_coin_center)

        except IndexError:
            pass
        continue
    monsters_pos = []
    # "monsters_pos" indicates the positions of all monsters
    for i in range(curr_screen.num_monsters):
        monster_image = curr_screen.ids['monster' +
                                        str(i+1)+'_image_lvl'+str(lvl_num)]
        monsters_pos.append([monster_image.pos_hint['x'],
                            monster_image.pos_hint['y']])
    fires_pos = []
    # "fires_pos" indicates the positions of all fires
    for i in range(curr_screen.num_fires):
        fire_image = curr_screen.ids['fire'+str(i+1)+'_lvl'+str(lvl_num)]
        fires_pos.append([fire_image.pos_hint['x'], fire_image.pos_hint['y']])

    return coin_pos, monsters_pos, fires_pos


class CointexApp(kivy.app.App):

    def _init_(self):
        self.start_time = None
        self.end_time = None

    def on_start(self):
        music_dir = os.getcwd()+"/music/"
        self.main_bg_music = kivy.core.audio.SoundLoader.load(
            music_dir+"bg_music_piano_flute.wav")
        self.main_bg_music.loop = True
        self.main_bg_music.volume = 0.1  # smaller the volume
        self.main_bg_music.play()

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

    def read_game_info(self):
        return 24, True

    def activate_levels(self, next_level_num, congrats_displayed_once):
        num_levels = len(
            self.root.screens[0].ids['lvls_imagebuttons'].children)

        levels_imagebuttons = self.root.screens[0].ids['lvls_imagebuttons'].children
        for i in range(num_levels-next_level_num, num_levels):
            levels_imagebuttons[i].disabled = False
            levels_imagebuttons[i].color = [1, 1, 1, 1]

        for i in range(0, num_levels-next_level_num):
            levels_imagebuttons[i].disabled = True
            levels_imagebuttons[i].color = [1, 1, 1, 0.5]

        if next_level_num == (num_levels+1) and congrats_displayed_once == False:
            self.root.current = "alllevelscompleted"

    def screen_on_pre_leave(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        for i in range(curr_screen.num_monsters):
            curr_screen.ids['monster'+str(i+1)+'_image_lvl' +
                            str(screen_num)].pos_hint = {'x': 0.8, 'y': 0.8}
        curr_screen.ids['character_image_lvl' +
                        str(screen_num)].pos_hint = {'x': 0.0, 'y': 0.0}

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

    def screen_on_pre_enter(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        curr_screen.character_killed = False
        curr_screen.num_coins_collected = 0
        curr_screen.ids['character_image_lvl'+str(
            screen_num)].im_num = curr_screen.ids['character_image_lvl'+str(screen_num)].start_im_num
        for i in range(curr_screen.num_monsters):
            curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(
                screen_num)].im_num = curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].start_im_num
        curr_screen.ids['num_coins_collected_lvl' +
                        str(screen_num)].text = "Coins 0/"+str(curr_screen.num_coins)
        curr_screen.ids['level_number_lvl' +
                        str(screen_num)].text = "Level "+str(screen_num)

        curr_screen.num_collisions_hit = 0
        remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(
            screen_num)]
        remaining_life_percent_lvl_widget.size_hint = (
            remaining_life_percent_lvl_widget.remaining_life_size_hint_x, remaining_life_percent_lvl_widget.size_hint[1])

        for i in range(curr_screen.num_fires):
            curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)
                            ].pos_hint = {'x': 1.1, 'y': 1.1}

        for key, coin in curr_screen.coins_ids.items():
            curr_screen.ids['layout_lvl'+str(screen_num)].remove_widget(coin)
        curr_screen.coins_ids = {}

        coin_width = 0.05
        coin_height = 0.05

        curr_screen = self.root.screens[screen_num]

        section_width = 1.0/curr_screen.num_coins

        for k in range(curr_screen.num_coins):
            x = random.uniform(section_width*k, section_width*(k+1)-coin_width)
            y = random.uniform(0, 1-coin_height)
            coin = kivy.uix.image.Image(source="other-images/coin.png", size_hint=(
                coin_width, coin_height), pos_hint={'x': x, 'y': y}, allow_stretch=True)
            Rectangle(pos=coin.pos, size=(1, 1))

            curr_screen.ids['layout_lvl' +
                            str(screen_num)].add_widget(coin, index=-1)
            curr_screen.coins_ids['coin'+str(k)] = coin

    def screen_on_enter(self, screen_num):
        self.start_time = time.time()
        music_dir = os.getcwd()+"/music/"
        self.bg_music = kivy.core.audio.SoundLoader.load(
            music_dir+"bg_music_piano.wav")
        # start to count the time when the agent is acting:
        time.perf_counter()
        self.bg_music.loop = True

        self.coin_sound = kivy.core.audio.SoundLoader.load(
            music_dir+"coin.wav")
        self.level_completed_sound = kivy.core.audio.SoundLoader.load(
            music_dir+"level_completed_flaute.wav")
        self.char_death_sound = kivy.core.audio.SoundLoader.load(
            music_dir+"char_death_flaute.wav")

        self.bg_music.volume = 0.1  # smaller the volume
        self.coin_sound.volume = 0.1  # smaller the volume
        self.level_completed_sound.volume = 0.1  # smaller the volume
        self.char_death_sound.volume = 0.1  # smaller the volume

        self.bg_music.play()

        curr_screen = self.root.screens[screen_num]
        for i in range(curr_screen.num_monsters):
            monster_image = curr_screen.ids['monster' +
                                            str(i+1)+'_image_lvl'+str(screen_num)]
            new_pos = (random.uniform(
                0.0, 1 - monster_image.size_hint[0]/4), random.uniform(0.0, 1 - monster_image.size_hint[1]/4))
            self.start_monst_animation(monster_image=monster_image, new_pos=new_pos, anim_duration=random.uniform(
                monster_image.monst_anim_duration_low, monster_image.monst_anim_duration_high))

        for i in range(curr_screen.num_fires):
            fire_widget = curr_screen.ids['fire' +
                                          str(i+1)+'_lvl'+str(screen_num)]
            self.start_fire_animation(
                fire_widget=fire_widget, pos=(0.0, 0.5), anim_duration=5.0)

        global lvl_num

        lvl_num = screen_num

        collectCoinThread = CollectCoinThread(screen=curr_screen)
        collectCoinThread.start()

    def start_monst_animation(self, monster_image, new_pos, anim_duration):
        monst_anim = kivy.animation.Animation(pos_hint={
                                              'x': new_pos[0], 'y': new_pos[1]}, im_num=monster_image.end_im_num, duration=anim_duration)
        monst_anim.bind(on_complete=self.monst_animation_completed)
        monst_anim.start(monster_image)

    def monst_animation_completed(self, *args):
        monster_image = args[1]
        monster_image.im_num = monster_image.start_im_num

        new_pos = (random.uniform(
            0.0, 1 - monster_image.size_hint[0]/4), random.uniform(0.0, 1 - monster_image.size_hint[1]/4))
        self.start_monst_animation(monster_image=monster_image, new_pos=new_pos, anim_duration=random.uniform(
            monster_image.monst_anim_duration_low, monster_image.monst_anim_duration_high))

    def monst_pos_hint(self, monster_image):
        screen_num = int(monster_image.parent.parent.name[5:])
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl' +
                                          str(screen_num)]

        character_center = character_image.center
        monster_center = monster_image.center

        gab_x = character_image.width / 2
        gab_y = character_image.height / 2
        if character_image.collide_widget(monster_image) and abs(character_center[0] - monster_center[0]) <= gab_x and abs(character_center[1] - monster_center[1]) <= gab_y:
            curr_screen.num_collisions_hit = curr_screen.num_collisions_hit + 1
            life_percent = float(curr_screen.num_collisions_hit) / \
                float(curr_screen.num_collisions_level)

#            life_remaining_percent = 100-round(life_percent, 2)*100
#            remaining_life_percent_lvl_widget.text = str(int(life_remaining_percent))+"%"
            remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(
                screen_num)]
            remaining_life_size_hint_x = remaining_life_percent_lvl_widget.remaining_life_size_hint_x
            remaining_life_percent_lvl_widget.size_hint = (
                remaining_life_size_hint_x-remaining_life_size_hint_x*life_percent, remaining_life_percent_lvl_widget.size_hint[1])

            if curr_screen.num_collisions_hit == curr_screen.num_collisions_level:
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters):
                    kivy.animation.Animation.cancel_all(
                        curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires):
                    kivy.animation.Animation.cancel_all(
                        curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(
                    im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(
                    self.back_to_main_screen, curr_screen.parent), 3)

    def change_monst_im(self, monster_image):
        monster_image.source = "monsters-images/" + \
            str(int(monster_image.im_num)) + ".png"

    def touch_down_handler(self, screen_num, args):
        curr_screen = self.root.screens[screen_num]
        if curr_screen.character_killed == False:
            self.start_char_animation(screen_num, args[1].spos)

    def start_char_animation(self, screen_num, touch_pos):
        curr_screen = self.root.screens[screen_num]
        if curr_screen.character_killed:
            return

        character_image = curr_screen.ids['character_image_lvl' +
                                          str(screen_num)]
        character_image.im_num = character_image.start_im_num
        char_anim = kivy.animation.Animation(pos_hint={'x': touch_pos[0] - character_image.size_hint[0] / 2, 'y': touch_pos[1] -
                                             character_image.size_hint[1] / 2}, im_num=character_image.end_im_num, duration=curr_screen.char_anim_duration)
        char_anim.bind(on_complete=self.char_animation_completed)
        char_anim.start(character_image)

    def char_animation_completed(self, *args):
        character_image = args[1]
        character_image.im_num = character_image.start_im_num

    def char_pos_hint(self, character_image):
        screen_num = int(character_image.parent.parent.name[5:])
        character_center = character_image.center

        gab_x = character_image.width / 2
        gab_y = character_image.height / 2
        coins_to_delete = []
        curr_screen = self.root.screens[screen_num]

        for coin_key, curr_coin in curr_screen.coins_ids.items():
            curr_coin_center = curr_coin.center
            if character_image.collide_widget(curr_coin) and abs(character_center[0] - curr_coin_center[0]) <= gab_x and abs(character_center[1] - curr_coin_center[1]) <= gab_y:
                self.coin_sound.play()
                coins_to_delete.append(coin_key)
                curr_screen.ids['layout_lvl' +
                                str(screen_num)].remove_widget(curr_coin)
                curr_screen.num_coins_collected = curr_screen.num_coins_collected + 1
                curr_screen.ids['num_coins_collected_lvl'+str(screen_num)].text = "Coins "+str(
                    curr_screen.num_coins_collected)+"/"+str(curr_screen.num_coins)
                if curr_screen.num_coins_collected == curr_screen.num_coins:
                    self.end_time = time.time()
                    print(f"execution time: {round(self.end_time - self.start_time, 4)}")
                    self.bg_music.stop()
                    self.level_completed_sound.play()
                    kivy.clock.Clock.schedule_once(functools.partial(
                        self.back_to_main_screen, curr_screen.parent), 3)
                    for i in range(curr_screen.num_monsters):
                        kivy.animation.Animation.cancel_all(
                            curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                    for i in range(curr_screen.num_fires):
                        kivy.animation.Animation.cancel_all(
                            curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                    next_level_num, congrats_displayed_once = self.read_game_info()
                    if (screen_num+1) > next_level_num:
                        game_info_file = open("game_info", 'wb')
                        pickle.dump(
                            [{'lastlvl': screen_num+1, "congrats_displayed_once": False}], game_info_file)
                        game_info_file.close()
                    else:
                        game_info_file = open("game_info", 'wb')
                        pickle.dump(
                            [{'lastlvl': next_level_num, "congrats_displayed_once": True}], game_info_file)
                        game_info_file.close()

        if len(coins_to_delete) > 0:
            for coin_key in coins_to_delete:
                del curr_screen.coins_ids[coin_key]

    def change_char_im(self, character_image):
        character_image.source = "character-images/" + \
            str(int(character_image.im_num)) + ".png"

    def start_fire_animation(self, fire_widget, pos, anim_duration):
        fire_anim = kivy.animation.Animation(pos_hint=fire_widget.fire_start_pos_hint, duration=fire_widget.fire_anim_duration) + \
            kivy.animation.Animation(
                pos_hint=fire_widget.fire_end_pos_hint, duration=fire_widget.fire_anim_duration)
        fire_anim.repeat = True
        fire_anim.start(fire_widget)

    def fire_pos_hint(self, fire_widget):
        screen_num = int(fire_widget.parent.parent.name[5:])
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl' +
                                          str(screen_num)]

        character_center = character_image.center
        fire_center = fire_widget.center

        gab_x = character_image.width / 3
        gab_y = character_image.height / 3
        if character_image.collide_widget(fire_widget) and abs(character_center[0] - fire_center[0]) <= gab_x and abs(character_center[1] - fire_center[1]) <= gab_y:
            curr_screen.num_collisions_hit = curr_screen.num_collisions_hit + 1
            life_percent = float(curr_screen.num_collisions_hit) / \
                float(curr_screen.num_collisions_level)

            remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(
                screen_num)]
#            life_remaining_percent = 100-round(life_percent, 2)*100
#            remaining_life_percent_lvl_widget.text = str(int(life_remaining_percent))+"%"

            remaining_life_size_hint_x = remaining_life_percent_lvl_widget.remaining_life_size_hint_x
            remaining_life_percent_lvl_widget.size_hint = (
                remaining_life_size_hint_x-remaining_life_size_hint_x*life_percent, remaining_life_percent_lvl_widget.size_hint[1])

            if curr_screen.num_collisions_hit == curr_screen.num_collisions_level:
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters):
                    kivy.animation.Animation.cancel_all(
                        curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires):
                    kivy.animation.Animation.cancel_all(
                        curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(
                    im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(
                    self.back_to_main_screen, curr_screen.parent), 3)

    def damage_check(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        character_img = curr_screen.ids["character_image_lvl" +
                                        str(screen_num)]

        for monster in range(curr_screen.num_monsters):
            monster_img = curr_screen.ids["monster" +
                                          str(monster + 1) + "_image_lvl" + str(screen_num)]

            character_center = character_img.center
            monster_center = monster_img.center

            gab_x = character_img.width * 1.5
            gab_y = character_img.height * 1.5

            if character_img.collide_widget(monster_img) and abs(character_center[0] - monster_center[0]) <= gab_x and abs(character_center[1] - monster_center[1]) <= gab_y:
                return True

        for fire in range(curr_screen.num_fires):

            fire_widget = curr_screen.ids["fire" +
                                          str(fire + 1) + "_lvl" + str(screen_num)]
            fire_center = fire_widget.center

            character_center = character_img.center

            gab_x = character_img.width
            gab_y = character_img.height

            if character_img.collide_widget(fire_widget) and abs(character_center[0] - fire_center[0]) <= gab_x and abs(character_center[1] - fire_center[1]) <= gab_y:
                return True

        return False

    def get_player_health(self, screen_num):
        curr_screen = self.root.screens[screen_num]

        player_health = 1 - (float(curr_screen.num_collisions_hit) /
                             float(curr_screen.num_collisions_level))

        return player_health

    def back_to_main_screen(self, screenManager, *args):
        screenManager.current = "main"

    def main_screen_on_enter(self):
        self.main_bg_music.play()

    def main_screen_on_leave(self):
        self.main_bg_music.stop()


class ImageButton(kivy.uix.behaviors.ButtonBehavior, kivy.uix.image.Image):
    pass


class MainScreen(kivy.uix.screenmanager.Screen):
    pass


class AboutUs(kivy.uix.screenmanager.Screen):
    pass


class AllLevelsCompleted(kivy.uix.screenmanager.Screen):
    pass


class Level1(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 5
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 20


class Level2(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 8
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30


class Level3(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30


class Level4(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 20


class Level5(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 20


class Level6(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 3
    num_collisions_hit = 0
    num_collisions_level = 20


class Level7(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25


class Level8(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25


class Level9(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25


class Level10(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 14
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30


class Level11(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 30


class Level12(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 30


class Level13(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 20


class Level14(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 6
    num_collisions_hit = 0
    num_collisions_level = 30


class Level15(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 16
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 3
    num_collisions_hit = 0
    num_collisions_level = 30


class Level16(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 35


class Level17(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30


class Level18(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30


class Level19(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 6
    num_collisions_hit = 0
    num_collisions_level = 30


class Level20(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 8
    num_collisions_hit = 0
    num_collisions_level = 30


class Level21(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 18
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30


class Level22(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 20
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30


class Level23(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 25
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 30


class Level24(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 20
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 30


app = CointexApp()
app.title = "CoinTex"
app.icon = 'cointex_logo.png'
# initialize the thread pool and create the runner

app.run()
