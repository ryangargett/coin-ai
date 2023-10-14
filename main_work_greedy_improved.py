import kivy.app
import kivy.uix.screenmanager
import kivy.uix.image
import random
import kivy.core.audio
import os
import functools
import kivy.uix.behaviors
import pickle
import pygad
import threading
import kivy.base
import random
import time
from typing import DefaultDict

import numpy as np

import networkx as nx
from networkx.utils import pairwise

import math

INT_MAX = 2147483647


class CollectCoinThread(threading.Thread):

    def __init__(self, screen):
        super().__init__()
        self.screen = screen
        self.MONSTER_PENALTY = 1.2
        self.FIRE_PENALTY = 1.5
        self.SAFETY_THRESHOLD = 0.2
        self.PLAYER_HEALTH_THRESHOLD = 0.7

    def create_tsp_matrix(self, start_pos):

        coin_pos, monster_pos, fire_pos = position_func_v3()

        num_nodes = len(coin_pos) + 1
        tsp_matrix = [[0] * num_nodes for _ in range(num_nodes)]

        # ii and jj are the current and potential future positions of the player character

        for ii in range(num_nodes):
            for jj in range(num_nodes):
                if ii == 0:
                    if jj > 0:
                        tsp_matrix[ii][jj] = self._get_euclidean_distance(
                            start_pos, coin_pos[jj-1])
                else:
                    tsp_matrix[ii][jj] = self._get_euclidean_distance(
                        coin_pos[ii-1], coin_pos[jj-1])
                    tsp_matrix[ii][jj] += self.get_weighted_penalty(
                        coin_pos[jj-1], monster_pos, fire_pos)

        return tsp_matrix

    def get_minimum_route(self, tsp_matrix):
        '''
        use Nearest Neighbours to calculate minimum route
        '''

        coin_pos, monster_pos, fire_pos = position_func_v3()

        num_nodes = len(tsp_matrix)

        # set all nodes to be unvisited initially
        visited = [False] * num_nodes

        route = [0]
        visited[0] = True

        for _ in range(num_nodes - 1):
            curr_node = route[-1]
            nearest_node = None
            min_cost = float("inf")

            for ii in range(num_nodes):
                if not visited[ii]:
                    distance = tsp_matrix[curr_node][ii]
                    weighted_penalty = self.get_weighted_penalty(
                        coin_pos[ii-1], monster_pos, fire_pos)

                    total_cost = distance + weighted_penalty

                    if total_cost < min_cost:
                        nearest_node = ii
                        min_cost = total_cost

            if nearest_node is not None:

                route.append(nearest_node)
                visited[nearest_node] = True

        return route

    def _get_euclidean_distance(self, pos1, pos2):
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance

    def _get_penalty(self, threat_pos, coin_pos, threat_type):

        total_penalty = 0

        for threat in threat_pos:
            distance = self._get_euclidean_distance(coin_pos, threat)

            if threat_type == "fire":
                penalty = 5 / (distance + 1e-5)  # avoid division by zero
            else:
                penalty = 2 / (distance + 1e-5)
            total_penalty += penalty

        return total_penalty

    def get_weighted_penalty(self, curr_pos, monster_pos, fire_pos):

        fire_penalty = 0

        monster_penalty = self._get_penalty(
            monster_pos, curr_pos, threat_type="monster")

        if fire_pos is not None:
            fire_penalty = self._get_penalty(
                fire_pos, curr_pos, threat_type="fire")

        weighted_penalty = self.MONSTER_PENALTY * \
            monster_penalty + self.FIRE_PENALTY * fire_penalty

        return weighted_penalty

    def get_safe_pos(self, start_pos):

        MOVE_FACTOR = 0.05

        _, monster_pos, fire_pos = position_func_v3()

        candidate_pos = [
            (start_pos[0] + MOVE_FACTOR, start_pos[1]),
            (start_pos[0] - MOVE_FACTOR, start_pos[1]),
            (start_pos[0] + MOVE_FACTOR, start_pos[1] + MOVE_FACTOR),
            (start_pos[0] + MOVE_FACTOR, start_pos[1] - MOVE_FACTOR),
            (start_pos[0] - MOVE_FACTOR, start_pos[1] + MOVE_FACTOR),
            (start_pos[0] - MOVE_FACTOR, start_pos[1] - MOVE_FACTOR),
            (start_pos[0], start_pos[1] + MOVE_FACTOR),
            (start_pos[0], start_pos[1] - MOVE_FACTOR)
        ]

        min_penalty = float("inf")
        safe_position = start_pos

        for pos in candidate_pos:
            weighted_penalty = self.get_weighted_penalty(
                pos, monster_pos, fire_pos)

            if weighted_penalty < min_penalty:
                if pos[0] >= 0 and pos[0] <= 1 and pos[1] >= 0 and pos[1] <= 1:
                    min_penalty = weighted_penalty
                    safe_position = pos

        return safe_position

    def run(self):
        start_time = time.time()
        start_pos = [0.1, 0.1]
        app.start_char_animation(lvl_num, start_pos)
        coin_pos, monster_pos, fire_pos = position_func_v3()
        num_coins = len(coin_pos)
        tsp = self.create_tsp_matrix(
            start_pos=start_pos)
        route = self.get_minimum_route(tsp)

        init_coins = num_coins

        while num_coins > 0:

            risk_factor = 0.1

            move_made = False

            next_coin_idx = route[1] - 1
            next_coin_pos = coin_pos[next_coin_idx]

            if num_coins <= init_coins * self.SAFETY_THRESHOLD:
                safety_threshold = 0.8
            else:
                safety_threshold = 0.3

            remaining_health = app.get_player_health(lvl_num)

            if remaining_health <= self.PLAYER_HEALTH_THRESHOLD:
                risk_factor = remaining_health
            else:
                risk_factor = 1.5

            stalling_penalty = 0
            stall_counter = 0

            while move_made == False:

                if app.damage_check(lvl_num) == True:
                    print(f"before hit: {start_pos}")
                    new_pos = self.get_safe_pos(start_pos)
                    print("after hit: ", new_pos)
                    app.start_char_animation(lvl_num, new_pos)
                    start_pos = new_pos

                tsp = self.create_tsp_matrix(
                    start_pos=start_pos)
                route = self.get_minimum_route(
                    tsp)
                next_coin_idx = route[1] - 1
                next_coin_pos = coin_pos[next_coin_idx]

                coin_pos, monster_pos, fire_pos = position_func_v3()

                weighted_curr_threat = self.get_weighted_penalty(
                    start_pos, monster_pos, fire_pos)
                weighted_future_threat = self.get_weighted_penalty(
                    next_coin_pos, monster_pos, fire_pos)

                if weighted_future_threat * safety_threshold < weighted_curr_threat + stalling_penalty:
                    move_made = True

                if stall_counter > 1:
                    print(f"stall counter: {stall_counter}")
                    time.sleep(0.025)

                stalling_penalty += risk_factor

                stall_counter += 1

            app.start_char_animation(lvl_num, next_coin_pos)
            time.sleep(0.3)

            coin_pos, monster_pos, fire_pos = position_func_v3()

            num_coins = len(coin_pos)

            tsp = self.create_tsp_matrix(
                next_coin_pos)
            route = self.get_minimum_route(
                tsp)

            start_pos = next_coin_pos

        end_time = time.time()
        print(f"execution time: {end_time - start_time}")


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
            curr_screen.ids['layout_lvl' +
                            str(screen_num)].add_widget(coin, index=-1)
            curr_screen.coins_ids['coin'+str(k)] = coin

    def screen_on_enter(self, screen_num):
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

        gab_x = character_image.width * 2
        gab_y = character_image.height * 2
        coins_to_delete = []
        curr_screen = self.root.screens[screen_num]

        for coin_key, curr_coin in curr_screen.coins_ids.items():
            curr_coin_center = curr_coin.center
            if character_image.collide_widget(curr_coin) and abs(character_center[0] - curr_coin_center[0]) <= gab_x / 2 and abs(character_center[1] - curr_coin_center[1]) <= gab_y / 2:
                self.coin_sound.play()
                coins_to_delete.append(coin_key)
                curr_screen.ids['layout_lvl' +
                                str(screen_num)].remove_widget(curr_coin)
                curr_screen.num_coins_collected = curr_screen.num_coins_collected + 1
                curr_screen.ids['num_coins_collected_lvl'+str(screen_num)].text = "Coins "+str(
                    curr_screen.num_coins_collected)+"/"+str(curr_screen.num_coins)
                if curr_screen.num_coins_collected == curr_screen.num_coins:
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
