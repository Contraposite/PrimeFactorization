#!/usr/bin/env python
# coding: utf-8

# In[1]:


#todo

#add sounds - try pyjnius


# # normal module imports

# In[2]:


import random
import pickle
import requests
import json


# # firebase import

# In[3]:


import pyrebase
firebase_config = {
    'apiKey': 'AIzaSyBIaPRIXT-XtCKNoF33Bi229TEF1O4oW3U',
    'authDomain': 'prime-factorizer.firebaseapp.com',
    'databaseURL': 'https://prime-factorizer-default-rtdb.europe-west1.firebasedatabase.app/',
    'storageBucket': 'gs://prime-factorizer.appspot.com'
}

firebase = pyrebase.initialize_app(firebase_config)
try:
    print(auth.current_user['localId'])
except Exception:
    auth = firebase.auth()
db = firebase.database()

UID = None #set later
token = None #set later

#####
# handy functions for firebase
#####

#data_test = {"score": 5, 'time': {'12':True}}

#set (update)
#db.child('player1').update(data_test)

#get
#db.child('player1').child('score').get().val()

#remove
#db.child('player1').child('score').remove()

#iterate
#toy_box_location = db.child('toy box')
#for toy in toy_box_location.each():
#    print(toy.val())
#    print(toy.key())


# # kivy imports

# In[4]:


from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.gridlayout import GridLayout
from kivy.properties import NumericProperty, ObjectProperty, ListProperty
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.lang import Builder
from kivy_garden.graph import LinePlot
from kivy_garden.graph import ScatterPlot
#from kivy.uix.image import Image
#from kivy.core.audio import SoundLoader


# # firebase functions

# In[5]:


#log in / sign up

def log_in(email, password, signing_up=False):
    
    global user
    global UID
    global token
    
    user = None
    UID = None
    token = None
    
    auth.current_user = None #sign out if already signed in
    
    if signing_up:
        try:
            auth.create_user_with_email_and_password(email, password)
            user = auth.sign_in_with_email_and_password(email, password)
            logged_in = True
        except Exception as ex:
            print(ex)

    else:
        #try to log in existing user
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            logged_in = True
        except Exception as ex:
            print(ex)

    try:
        UID = user['localId']
        token = user['idToken']
        set_friendly_user_id()
        return True #successful
    except Exception:
        pass
    
    return False #unsuccessful


# In[6]:


def download_data():
    global data
    try:
        db_data = db.child('play data').child(UID).get(token).val()
        if db_data['games_played'] >= data['games_played']: #if firebase data is not outdated
            data = db_data
            print('data downloaded from firebase')
    except Exception as ex:
        print('data download unsuccessful')
        print(ex)        


# In[7]:


def upload_data():
    download_data() #if we're out of date, need to update to the firebase data now.
    try:
        db_data = db.child('play data').child(UID).get(token).val()
        if (db_data == None) or (data['games_played'] >= db_data['games_played']): #if local data is not outdated
            db.child('play data').update({UID:data}, token)
            db.child('best games leaderboard').update({UID:data['saved_games']['best_game']}, token)
            print('data uploaded to firebase')
        else:
            print(f'games played local: {data["games_played"]}, games played firebase: {db_data["games_played"]}')
    except Exception as ex:
        print('data upload unsuccessful')
        print(ex)


# In[8]:


def get_best_games_leaderboard():
    try:
        return db.child('best games leaderboard').get(token).val()
    except Exception as ex:
        print(ex)


# In[9]:


def set_friendly_user_id():
    try:
        if db.child('user public data').child(UID).child('friendly ID').get(token).val() == None: # only if I don't have a friendly ID
            print( [ user for user in db.child('user public data').get(token).val().keys() ] )
            existing_ids = [ db.child('user public data').child(user).child('friendly ID').get(token).val() for user in db.child('user public data').get(token).val().keys() ]
            if None in existing_ids:
                existing_ids.remove(None)
            print(f'existing ids: {existing_ids}')
            largest_id = max(existing_ids)
            my_new_id = largest_id +1
            #db.child('user public data').update({UID:{'friendly ID':my_new_id}}, token)
            db.child('user public data').child(UID).update({'friendly ID':my_new_id}, token)
    except Exception as ex:
        print(ex)


# In[10]:


def get_friendly_user_id():
    try:
        set_friendly_user_id()
        return db.child('user public data').child(UID).child('friendly ID').get(token).val()
    except Exception:
        return None


# # functions

# In[11]:


def level_up_sound():
    pass
    #sound = SoundLoader.load('point.wav')
    #if sound:
    #    #sound.play()
    #    pass
        
#tap_wav = SoundLoader.load('tap.wav')
def tap_sound():
    pass
    #tap_wav.play()
    #sound = SoundLoader.load('tap.wav')
    #if sound:
    #    sound.play()


# In[12]:


def save(obj):
    try:
        with open("../save_data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('data saved!')
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


# In[13]:


def load(filename):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            print('data loaded!')
            #for key in data:
                #print('\n***'+key)
                #print(data[key])
            return data
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


# In[14]:


#get/create the dictionary which will be the saved object storing important data
data = load('../save_data.pickle')
if data == None:
    data = {'saved_games':{'best_game': None, 'last_game':None}, 'best_times_per_score':[0], 'games_played':0, 'average_score':0, 'target_history': {'0':{'target':0, 'wins':0, 'losses':0, 'avg_time':0}}}
    


# In[15]:


def target_data_sorted_by(prop):
    listed_data = [data['target_history'][key] for key in data['target_history']]
    listed_data.sort( key=lambda target_data: target_data[prop])
    return listed_data


# In[16]:


def record_win(target_number, time_taken, game_time, game_type):
    #record data for later use on 'game finished' graph
    data['saved_games']['last_game'].append( {'target': target_number, 'game_time': game_time} )
    #ignore function call if the game type is training
    if game_type == 'training':
        return
    #make sure a dictionary exists for this target
    if not str(target_number) in data['target_history']:
        data['target_history'][str(target_number)] = {'target':target_number, 'wins':0, 'losses':0, 'avg_time':0}
    #modify values
    avg_before = data['target_history'][str(target_number)]['avg_time']
    wins_before = data['target_history'][str(target_number)]['wins']
    data['target_history'][str(target_number)]['wins'] += 1 #increment the number of wins for this target
    wins_after = data['target_history'][str(target_number)]['wins']
    data['target_history'][str(target_number)]['avg_time'] = (avg_before*wins_before + time_taken) / wins_after #set the average (finds the total time taken and divides by total wins)
    score = len(data['saved_games']['last_game'])
    if len(data['best_times_per_score'])<score+1:
        data['best_times_per_score'].append(game_time)
    elif data['best_times_per_score'][score]>game_time:
        data['best_times_per_score'][score] = game_time
        


# In[17]:


def record_loss(target_number, game_type):
    #ignore function call if the game type is training
    if game_type == 'training':
        return
    #make sure a dictionary exists for this target
    if not str(target_number) in data['target_history']:
        data['target_history'][str(target_number)] = {'target':target_number, 'wins':0, 'losses':0, 'avg_time':0}
    #modify values
    data['target_history'][str(target_number)]['losses'] += 1 #increment the number of losses for this target


# In[18]:


#this function is not mine
def get_factors(n):
    prime_factors = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            prime_factors.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        prime_factors.append(n)
    return prime_factors


# In[19]:


def graph_tick_dist(val):
    steps = 6
    if val <5:
        return 1
    elif val < 25:
        return 2* max(int(val/steps/2), 1)
    elif val < 100:
        return 5* max(int(val/steps/5), 1)
    elif val < 200:
        return 10* max(int(val/steps/10), 1)
    else:
        return 25* max(int(val/steps/25), 1)

#for i in range(10):
#    num=2**i
#    print(num, graph_tick_dist(num))


# # global data

# In[20]:


keycodes = {
    49:'1',
    257:'1',
    50:'2',
    258:'2',
    51:'3',
    259:'3',
    52:'4',
    260:'4',
    53:'5',
    261:'5',
    54:'6',
    262:'6',
    55:'7',
    263:'7',
    56:'8',
    264:'8',
    57:'9',
    265:'9',
    48:'0',
    256:'0',
    
}


# # classes

# In[21]:


class ThemeManager:
    
    def __init__(self):
        self.primary = (0.282,0.270,0.288,1)       
        self.secondary = (0,0.7,1,1)
        self.background = (0.092,0.085,0.095,1)
        self.tertiary = (0.9,0.525,0.525,1)
        Window.clearcolor = self.background
        
    def dark(self, colour):
        return [0.5*val for val in colour[:-1]] + [colour[-1]]        
    def light(self, colour):
        return [1-0.5*(1-val) for val in colour[:-1]] + [colour[-1]]
    def mix(self,c1,c2):
        return [(c1[i]+c2[i])/2 for i in range(4)]


# In[22]:


# I think I could have actually done this just in the .kv file...
class MyWindowManager(ScreenManager):
    pass
class TextLabel(Label):
    pass
class MultilineTextLabel(Label):
    pass
class FactorBox(TextLabel):
    pass
class BackgroundTextLabel(TextLabel):
    pass
class LeaderboardTextLabel(TextLabel):
    pass


# In[23]:


class MyGrid(Screen):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
        self.game_type = 'normal'
        self.ids.mode_btn.text = 'norm'
        self.ids.mode_btn.color = (0.5,0.5,0.5,1)
        self.reset()
        
        self.target_animation_stage = 0 #animiation not running
        
        self.TTS_plot = LinePlot( color=app.theme.secondary, line_width=2 )
        self.ids.TTS_graph.add_plot(self.TTS_plot)
        self.ids.TTS_graph.x_ticks_major=5
        self.TTS_plot.points = []
        
        Clock.schedule_interval(self.update_graphs, 1/15)
        
        self.training_set = [] #-just want it to not be empty as that would cause errors
    
    def update_graphs(self, dt):
        if self.manager.current == 'game_screen':
            TTS_points_with_prediction = self.TTS_graph_points + [(self.score+1, self.question_time-self.time_remaining)]
            self.TTS_plot.points = TTS_points_with_prediction
            self.ids.TTS_graph.xmax = self.score+1
            self.ids.TTS_graph.ymax = self.longest_time_taken+1        
    
    def reset(self):
        #need to declare target number early since it's referenced by the function which sets its initial value.
        #its value is used as an exclusion from the random options, to avoid getting the same number multiple times in a row.
        #resetting it to zero in this function means at the start of the game there are no exclusions.
        self.target_number = 0
        
        self.score = 0
        self.question_time = 100
        self.time_remaining = 100
        self.game_time = 0
        self.TTS_graph_points = [(0,0)]
        self.longest_time_taken = 0
        #- self.ST_graph_points = [(0,0)]
        data['saved_games']['last_game']=[]
        self.playing = False
        self.new_challenge(0)
        
        #reset button state (remove hints from last game)
        for btn in ['1','2','3','4','5','6','7','8','9','0']:
            self.ids[btn].state = 'normal'

    def new_challenge(self, game_time):
        self.answer = ""
        self.ids.running_input.text = ""
        self.question_time = 10
        self.time_remaining = self.question_time
        self.target_number = self.new_target(game_time)
        self.ids.target.text = str(self.target_number)
        self.factors_so_far = []
        self.ids.factor_stack.clear_widgets()
        self.factors_to_go = get_factors(self.target_number)
        self.hint_timer = 0
        
    def new_target(self, game_time):
        if not self.playing:
            return 2 #first target of all games will be this number 
        else:
            exclusion = self.target_number
            if self.game_type == 'normal':
                options = list(range(int(3+2*game_time),int(13+2*game_time)))
            elif self.game_type == 'training':
                pool_size_moving = 5 #size of pool from which to choose random number while going through the training set
                pool_size_final = 10 #size of pool from which to choose random number once we've reached the end of the training set
                clip_start = max(0, min(self.score-1, len(self.training_set)-pool_size_final) )
                clip_end = min(self.score+pool_size_moving, len(self.training_set)) #min( clip_start+pool_size_moving, len(self.training_set))
                options = self.training_set.copy()[clip_start:clip_end] #create a copy of the training set to use, and crop
                #print(f'score: {self.score}, start: {clip_start}, end: {clip_end}')
            #to avoid getting the same target twice in a row, exclude it from the options.
            if exclusion in options:
                options.remove(exclusion)
            return random.choice(options)
        
    def submit(self, key=""):
        
        #if not on the game screen, don't submit anything
        if not self.manager.current == 'game_screen':
            return
        
        tap_sound()
        
        self.answer = self.answer + key
        
        submission_text = self.answer
        submission = int(submission_text)
        
        if self.factors_to_go[0] == submission:
            self.factors_to_go.remove(submission)
            self.factors_so_far.append(submission)
            factor_img = FactorBox(text=str(submission))
            self.ids.factor_stack.add_widget(factor_img)
            self.answer = ""
        elif submission_text in str(self.factors_to_go[0]) and str(self.factors_to_go[0]).index(submission_text)==0:
            pass #this is part way towards the correct answer. let them continue writing.
        else:
            record_loss(self.target_number, self.game_type)
            self.lose_game(submission_text)
        
        if len(self.factors_to_go) < 1: #if the player has listed all of the prime factors
            self.score += 1
            level_up_sound()
            
            time_taken = self.question_time-self.time_remaining
            self.TTS_graph_points.append((self.score, time_taken))
            if time_taken > self.longest_time_taken:
                self.longest_time_taken = time_taken
            
            record_win(self.target_number, time_taken, self.game_time, self.game_type)
            
            self.playing = True #start the timer after the first target has been successfully completed
            self.target_animation_stage = 0.0001 #start the animation
            self.ids.target_fx.text = self.ids.target.text
            
            self.new_challenge(self.game_time)
            
        self.ids.running_input.text = self.answer
                
    def update(self, dt):
        if (self.manager.current == 'game_screen') and self.playing:
            self.game_time += dt
            self.time_remaining -= dt
            if self.time_remaining <0:
                record_loss(self.target_number, self.game_type)
                self.lose_game()
                
            if self.score<5: #you will get hints until you reach this score
                self.hint_timer += dt
                if self.hint_timer > 3: #time between hints
                    next_factor = str( self.factors_to_go[0] )
                    next_input = next_factor[len(self.ids.running_input.text)]
                    self.ids[next_input].state = 'down'
                    self.hint_timer = 0
        
        self.ids.target_fx.pos = self.target_fx_pos(self.target_animation_stage)
        self.ids.target_fx.size = self.target_fx_size(self.target_animation_stage)
        self.ids.target_fx.opacity = self.target_fx_opacity(self.target_animation_stage)
        self.ids.target_fx.font_size = min( self.ids.target_fx.height*0.8, self.ids.target_fx.width/(0.1+0.6*len(self.ids.target_fx.text)) )
        if self.target_animation_stage > 0:
            self.target_animation_stage += 0.05
        if self.target_animation_stage >= 1:
            self.target_animation_stage = 0
                    
        self.ids.time_display.value = self.time_remaining / self.question_time
    
    def target_fx_pos(self, stage):
        return (self.ids.target.x+(self.width*0.6*stage), self.ids.target.y)
    
    def target_fx_size(self, stage):
        return ( self.ids.target.width, self.ids.target.height )
    
    def target_fx_opacity(self, stage):
        if stage == 0:
            return 0
        else:
            return 0.5* (1-stage)
    
    def cycle_mode(self):
        if self.game_type == 'normal':
            self.game_type = 'training'
            self.ids.mode_btn.text = 'train'
            self.ids.mode_btn.color = (1,0.9,0.0,1)
                        
            training_targets_data = [] #temporary list of target data for sorting before putting in the training set
            listed_data = [data['target_history'][key] for key in data['target_history']] #we will sort this in different orders then add targets to the training_targets_data list
            #use different fitness formulae to sort the targets and add to the training targets data list.
            #fitness formula starts with a bias towards targets with high losses, then changes to bias targets with bad win/loss ratios.
            loop_count = 4 #numbere of different fitness formulae to use
            for i in range(loop_count):
                #sort based on a fitness formula which changes for each iteration of the loop
                listed_data.sort( key=lambda target_data: target_data['wins']/(0.01+ target_data['losses']**(loop_count-i)) )
                #in the order of worst to best fitness score, add a few targets to the target data list, if they are valid and not already added
                targets_added=0
                for j in range(len(listed_data)):
                    this_target_data = listed_data[j]
                    if (not this_target_data in training_targets_data) and (this_target_data['target']>2):
                        training_targets_data.append(this_target_data)
                        targets_added+=1
                    if targets_added>=5: #only add a max of 5 targets per fitness formula
                        break
            #do a final sort of the targets so that the training set is in order easy-to-hard
            training_targets_data.sort( key=lambda target_data: target_data['wins']/(0.01+ target_data['losses']), reverse=True )
            #we don't need the data dicts, just the target numbers, so extract these to create the training set
            self.training_set = [target_data['target'] for target_data in training_targets_data]
            #training set needs to be at least 2 long so add some if there aren't enough
            if len(self.training_set) <2:
                self.training_set += [2,3]
            print('training set: ', self.training_set)
            
            if self.score>0:  # if you switch modes in the middle of a proper game, that is considered 'chickening out' of the current target number, and is recorded as a loss.
                record_loss(self.target_number, self.game_type)
                self.lose_game('[switch game mode button]')
            else: # but if we hadn't started the game yet, just reset the game with the new mode.
                self.reset()
        else:
            self.game_type = 'normal'
            self.ids.mode_btn.text = 'norm'
            self.ids.mode_btn.color = (0.5,0.5,0.5,1)
            self.reset()
    
    def lose_game(self, losing_attempt='None'):
        #set number of games and average score in data dict
        games_before = data['games_played']
        average_before = data['average_score']
        data['games_played']+=1
        games_after = data['games_played']
        data['average_score'] = (games_before*average_before+self.score) / games_after
        #switch screen and set info of the game finished screen
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_over_screen'
        self.manager.get_screen('game_over_screen').ids.last_game_score.text = str(self.score)
        self.manager.get_screen('game_over_screen').ids.info_to_display.target = str(self.target_number)
        self.manager.get_screen('game_over_screen').ids.info_to_display.factors_so_far = str(self.factors_so_far)
        self.manager.get_screen('game_over_screen').ids.info_to_display.user_in = str(losing_attempt)
        self.manager.get_screen('game_over_screen').ids.info_to_display.answer = str(get_factors(self.target_number))
        
        self.manager.screens[1].update_graphs(self.game_type)
        
        save(data) # save after updating game over graph because data is modified there
        upload_data() # to firebase
        
    def show_info(self):
        if self.score>0 and self.game_type=='normal':  # if you switch modes in the middle of a proper game, that is considered 'chickening out' of the current target number, and is recorded as a loss.
            record_loss(self.target_number, self.game_type)
            self.lose_game('[info button]')
        else:
            self.manager.transition.direction = 'right'
            self.manager.current = 'info_screen'
        
    def show_menu(self):
        if self.score>0 and self.game_type=='normal':  # if you switch modes in the middle of a proper game, that is considered 'chickening out' of the current target number, and is recorded as a loss.
            record_loss(self.target_number, self.game_type)
            self.lose_game('[menu button]')
        else:
            self.manager.transition.direction = 'right'
            self.manager.current = 'menu_screen'
        


# In[24]:


class GameOver(Screen):
    
    info_to_display = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.ids.ST_graph.y_grid=True
        self.ids.ST_graph.y_grid_label=True
        self.ids.ST_graph.y_ticks_major=5
        self.ids.ST_graph.x_grid=True
        self.ids.ST_graph.x_grid_label=True
        self.ids.ST_graph.x_ticks_major=10
        
        self.ST_plot_best = LinePlot( color=[1,0,0,1], line_width=1.5 )
        self.ids.ST_graph.add_plot(self.ST_plot_best)
        self.ST_plot_best.points = []
        
        self.ST_plot = LinePlot( color=[0,0,1,1], line_width=1.5 )
        self.ids.ST_graph.add_plot(self.ST_plot)
        self.ST_plot.points = []
        
    def update_graphs(self, game_type):
        self.ST_plot.points = [(0,0)]+[(pt['game_time'],score+1) for score,pt in enumerate(data['saved_games']['last_game'])]
        
        this_score = len(data['saved_games']['last_game'])
        if len(data['saved_games']['last_game']) > 0:
            this_run_time = data['saved_games']['last_game'][-1]['game_time']
        else:
            this_run_time = 0
            
        best_game_time = None #set this in the code below
        
        if game_type == 'normal':
            #check against saved high score
            if data['saved_games']['best_game'] == None:
                data['saved_games']['best_game'] = data['saved_games']['last_game']
            high_score = len(data['saved_games']['best_game'])
            if len(data['saved_games']['best_game']) > 0:
                best_game_time = data['saved_games']['best_game'][-1]['game_time']
            else:
                best_game_time = 0
            if (this_score > high_score)  or  ((this_score == high_score) and (this_run_time < best_game_time)): #if the last game is better than the best saved game
                data['saved_games']['best_game'] = data['saved_games']['last_game']
            self.ST_plot_best.points = [(0,0)]+[(pt['game_time'],score+1) for score,pt in enumerate(data['saved_games']['best_game'])]
        elif game_type == 'training':
            if data['saved_games']['best_game'] == None:
                self.ST_plot_best.points = [(0,0)]
                best_game_time = 0
                high_score = 0
            else:
                self.ST_plot_best.points = [(0,0)]+[(pt['game_time'],score+1) for score,pt in enumerate(data['saved_games']['best_game'])]
                if len(data['saved_games']['best_game'])>0:
                    best_game_time = data['saved_games']['best_game'][-1]['game_time']
                else:
                    best_game_time=0
                high_score = len(data['saved_games']['best_game'])
        
        self.ids.ST_graph.xmax = max(this_run_time, best_game_time)+1
        self.ids.ST_graph.x_ticks_major = graph_tick_dist(self.ids.ST_graph.xmax)
        self.ids.ST_graph.ymax = max(this_score, high_score)+1
        self.ids.ST_graph.y_ticks_major = graph_tick_dist(self.ids.ST_graph.ymax)
        
    def refresh_scatterplots(self):
        return
        
    def restart(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[25]:


class Info(Screen):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
    def search(self, link):
        import webbrowser
        try:
            webbrowser.open(link)
        except ex as Exception:
            print('error opening link')
            
    def restart(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[26]:


class Stats(Screen):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.ids.best_ST_graph.x_grid=True
        self.ids.best_ST_graph.x_grid_label=True
        self.ids.best_ST_graph.x_ticks_major=5
        self.ids.best_ST_graph.y_grid=True
        self.ids.best_ST_graph.y_grid_label=True
        self.ids.best_ST_graph.y_ticks_major=5
        self.ids.WRN_graph.x_grid=True
        self.ids.WRN_graph.x_grid_label=True
        self.ids.WRN_graph.x_ticks_major=5
        self.ids.WRN_graph.y_grid_label=True
        self.ids.WRN_graph.y_ticks_major=0.25
        self.ids.TN_graph.x_grid=True
        self.ids.TN_graph.x_grid_label=True
        self.ids.TN_graph.x_ticks_major=5
        self.ids.TN_graph.y_grid_label=True
        self.ids.TN_graph.y_ticks_major=2
        
        self.best_ST_plot = LinePlot( color=[1,0,0,1], line_width=1.5 )
        self.ids.best_ST_graph.add_plot(self.best_ST_plot)
        self.best_ST_plot.points = []
        
        self.WRN_plot = ScatterPlot( color=[1,1,0,1], point_size=self.width*0.015 )
        self.ids.WRN_graph.add_plot(self.WRN_plot)
        self.WRN_plot.points = []
        
        self.TN_plot = ScatterPlot( color=[1,1,0,1], point_size=self.width*0.015 )
        self.ids.TN_graph.add_plot(self.TN_plot)
        self.TN_plot.points = []
    
    def update_info(self, player_id=''):
        
        user_data = None
        if player_id=='':
            user_data = data
        else:
            try:
                public_info = db.child('user public data').get(token).val()
                for user_UID in public_info.keys():
                    if str(public_info[user_UID]['friendly ID']) == player_id:
                        user_data = db.child('play data').child(user_UID).get(token).val()
                        break
            except Exception as ex:
                print(ex)
                        
        if user_data != None:
        
            if not user_data['saved_games']['best_game'] == None:
                self.best_ST_plot.points = [(0,0)]+[(pt['game_time'],score+1) for score,pt in enumerate(user_data['saved_games']['best_game'])]
                self.ids['highscore'].score = str( len(user_data['saved_games']["best_game"]) )
                if len(user_data['saved_games']['best_game']) > 0:
                    self.ids['highscore'].time = f'{(user_data["saved_games"]["best_game"][-1]["game_time"]):.1f}'
                else:
                    self.ids['highscore'].time = '0.0'
            self.WRN_plot.points = []
            self.TN_plot.points = []
            self.ids.low_win_rate_targets.text = 'targets with low win rates:'
            self.ids.low_win_rate_targets.rows = 1


            #sort the data by win rate
            sorted_data = []
            for key in user_data['target_history'].keys():
                sorted_data.append({'target':key, 'data':user_data['target_history'][key]})
                sorted_data.sort( key=lambda keydata: (keydata['data']['wins']/(0.001+keydata['data']['wins']+keydata['data']['losses'])) )

            #iterate through data and add points to graphs as we go
            for i,target_data in enumerate(sorted_data):
                key = target_data['target']
                wins = user_data['target_history'][key]['wins']
                losses = user_data['target_history'][key]['losses']
                avg_time = user_data['target_history'][key]['avg_time']
                if (wins+losses)>3:
                    win_rate = wins/(wins+losses)
                    self.WRN_plot.points.append((int(key), win_rate))
                    if win_rate < 0.75:
                        self.ids.low_win_rate_targets.text += f'\n{key} ({win_rate:.2f})'
                        self.ids.low_win_rate_targets.rows +=1
                if avg_time != 0:
                    self.TN_plot.points.append((int(key), avg_time))

            self.refresh_graphs()

            if self.ids.low_win_rate_targets.rows==1:
                self.ids.low_win_rate_targets.rows +=1
                self.ids.low_win_rate_targets.text += '\n[None]'

        
    def refresh_graphs(self):
                
        if data['saved_games']['best_game']!=None and len(data['saved_games']['best_game'])>0:
            self.ids.best_ST_graph.xmax = max(pt[0] for pt in self.best_ST_plot.points)+1 +0.0001*random.random() #random number fixes an issue with kivy only displaying the graph if it is different from before
            self.ids.best_ST_graph.x_ticks_major = graph_tick_dist(self.ids.best_ST_graph.xmax)
            self.ids.best_ST_graph.ymax = max(pt[1] for pt in self.best_ST_plot.points)+1
            self.ids.best_ST_graph.y_ticks_major = graph_tick_dist(self.ids.best_ST_graph.ymax)
        if len(self.WRN_plot.points)>0:
            self.ids.WRN_graph.xmax = max(pt[0] for pt in self.WRN_plot.points)+1 +0.0001*random.random() #random number fixes an issue with kivy only displaying the graph if it is different from before
            self.ids.WRN_graph.x_ticks_major = graph_tick_dist(self.ids.WRN_graph.xmax)
            self.ids.WRN_graph.ymax = 1.05
        if len(self.TN_plot.points)>0:
            self.ids.TN_graph.xmax = max(pt[0] for pt in self.TN_plot.points)+1 +0.0001*random.random() #random number fixes an issue with kivy only displaying the graph if it is different from before
            self.ids.TN_graph.x_ticks_major = graph_tick_dist(self.ids.TN_graph.xmax)
            self.ids.TN_graph.ymax = max(pt[1] for pt in self.TN_plot.points)+1
    
    def refresh_scatterplots(self):
        self.refresh_graphs()
    
    def search_1(self, ind):
        try:
            result = round( self.ids.best_ST_graph.plots[0].points[int(ind)][0], 1 )
            return f'{ind} @ {result} sec'
        except Exception as ex:
            return 'N/A'
        
    def search_2(self, ind):
        result = None
        try:
            for point in self.ids.WRN_graph.plots[0].points:
                if point[0] == int(ind):
                    result = point[1]
                    break
            if result == None:
                return 'N/A'
            return f'{ind} win rate:{result*100: .0f}%'
                
        except Exception as ex:
            return 'N/A'
        
    def search_3(self, ind):
        result = None
        try:
            for point in self.ids.TN_graph.plots[0].points:
                if point[0] == int(ind):
                    result = point[1]
                    break
            if result == None:
                return 'N/A'
            return f'{ind} time:{result: .1f} sec'
                
        except Exception as ex:
            return 'N/A'
    
    def change_screen(self, screen_name):
        self.manager.transition.direction = 'right'
        self.manager.current = screen_name


# In[27]:


class Login(Screen):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def submit_login(self, email, password, signing_up=False):
        #log out if already logged in
        auth.current_user = None
        
        #attempt to log in / sign up
        log_in(email, password, signing_up)
        login_success = auth.current_user != None
        if login_success:
            download_data()
            upload_data()
            self.manager.transition.direction = 'up'
            self.manager.current = 'game_screen'
            self.manager.get_screen('game_screen').reset()
                        
    def change_screen(self, screen_name):
        self.manager.transition.direction = 'up'
        self.manager.current = screen_name


# In[28]:


class Profile(Screen):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def set_public_name(self, name):
        #attempt to set the user's public name
        try:
            db.child('user public data').child(UID).update({'name': name}, token)
        except Exception as ex:
            print(ex)
            
    def change_screen(self, screen_name):
        self.set_public_name(self.ids.public_name.text) #try to set current input before leaving this screen
        
        self.manager.transition.direction = 'right'
        self.manager.current = screen_name


# In[29]:


class Menu(Screen):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def change_screen(self, screen_name):
        self.manager.transition.direction = 'left'
        self.manager.current = screen_name
        if screen_name == 'game_screen':
            self.manager.get_screen('game_screen').reset()
        elif screen_name == 'leaderboard_screen':
            self.manager.get_screen('leaderboard_screen').set_up()
        elif screen_name == 'stats_screen':
            self.manager.get_screen('stats_screen').update_info()


# In[30]:


class Leaderboard(Screen):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def set_up(self):
        for wid in self.ids.leaderboard_grid.children[:-4]:
            self.ids.leaderboard_grid.remove_widget(wid)
        
        try:
            
            leaderboard_data = db.child('best games leaderboard').get(token).val()
            entries = []
            for player in leaderboard_data.keys():
                try:
                    friendly_id = db.child('user public data').child(player).child('friendly ID').get(token).val()
                    name = db.child('user public data').child(player).child('name').get(token).val()
                    score = len(leaderboard_data[player])
                    entries.append( {'friendly id': friendly_id, 'score': score, 'name':name} )
                except Exception:
                    pass

            entries.sort( key=lambda d: d['score'] , reverse=True)

            for i,entry in enumerate(entries):
                leaderboard_grid = self.ids.leaderboard_grid
                place_wid = LeaderboardTextLabel(text=str(i+1))
                name_wid = LeaderboardTextLabel(text=str(entry['name']))
                id_wid = LeaderboardTextLabel(text=str(entry['friendly id']))
                score_wid = LeaderboardTextLabel(text=str(entry['score']))
                leaderboard_grid.add_widget(place_wid)
                leaderboard_grid.add_widget(name_wid)
                leaderboard_grid.add_widget(id_wid)
                leaderboard_grid.add_widget(score_wid)
        
        except Exception:
            pass #probably just not signed in, or no internet connection
            
    def change_screen(self, screen_name):
        self.manager.transition.direction = 'right'
        self.manager.current = screen_name


# In[31]:


class PrimeFactorizerApp(App):
    
    theme = ObjectProperty()
    
    def __init__(self, **kwargs):
        self.theme = ThemeManager()
        super().__init__(**kwargs)
            
    def build(self):
        root = Builder.load_file('PrimeFactorizer_UI.kv')
        
        self.SM = MyWindowManager()
        
        my_grid = MyGrid()
        game_over = GameOver()
        info = Info()
        stats = Stats()
        login = Login()
        menu = Menu()
        leaderboard = Leaderboard()
        profile = Profile()
        Clock.schedule_interval(my_grid.update, 1.0/60.0)
        
        self.SM.add_widget(my_grid)
        self.SM.add_widget(game_over)
        self.SM.add_widget(info)
        self.SM.add_widget(stats)
        self.SM.add_widget(login)
        self.SM.add_widget(menu)
        self.SM.add_widget(leaderboard)
        self.SM.add_widget(profile)
        
        self.SM.current = 'login_screen'
                        
        Window.bind(on_key_down = self.key_input)
        Window.bind(on_request_close = self.end_func)
        
        return self.SM    
    
    def key_input(self, window ,keycode, scancode, codepoint, modifier):
        #fix default crash behaviour on back button press on android
        #print('keycode: ', keycode)
        if keycode==27: #back button on android, esc on windows
            return True
        #elif keycode==32: #space on windows
        #    self.end_func()
        elif keycode in keycodes:
            key = keycodes[keycode]
            self.SM.get_screen('game_screen').submit(key)
            return False
        else:
            return False
        
    def end_func(self, *args):
        self.stop()
        Window.close()
        
    def on_pause(self):
        return True
        
if __name__ == '__main__':
    app = PrimeFactorizerApp()
    app.run()


# In[ ]:




