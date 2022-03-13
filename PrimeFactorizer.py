#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from kivy.core.audio import SoundLoader


# In[2]:


def level_up_sound():
    sound = SoundLoader.load('point.wav')
    if sound:
        #sound.play()
        pass
        
tap_wav = SoundLoader.load('tap.wav')
def tap_sound():
    tap_wav.play()
    #sound = SoundLoader.load('tap.wav')
    #if sound:
    #    sound.play()


# In[3]:


import random
import pickle


# In[4]:


def save(obj):
    try:
        with open("save_data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('data saved!')
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


# In[5]:


def load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


# In[6]:


#get/create the dictionary which will be the saved object storing important data
data = load('save_data.pickle')
if data == None:
    data = {'best_run': None, 'target_history': {'0':{'wins':0, 'losses':0, 'avg_time':0}}}
    
#print(data)


# In[7]:


def record_win(target_number, time_taken, game_type):
    #ignore function call if the game type is training
    if game_type == 'training':
        return
    #make sure a dictionary exists for this target
    if not str(target_number) in data['target_history']:
        data['target_history'][str(target_number)] = {'wins':0, 'losses':0, 'avg_time':0}
    #modify values
    avg_before = data['target_history'][str(target_number)]['avg_time']
    wins_before = data['target_history'][str(target_number)]['wins']
    data['target_history'][str(target_number)]['wins'] += 1 #increment the number of wins for this target
    wins_after = data['target_history'][str(target_number)]['wins']
    data['target_history'][str(target_number)]['avg_time'] = (avg_before*wins_before + time_taken) / wins_after #set the average (finds the total time taken and divides by total wins)


# In[8]:


def record_loss(target_number, game_type):
    #ignore function call if the game type is training
    if game_type == 'training':
        return
    #make sure a dictionary exists for this target
    if not str(target_number) in data['target_history']:
        data['target_history'][str(target_number)] = {'wins':0, 'losses':0, 'avg_time':0}
    #modify values
    data['target_history'][str(target_number)]['losses'] += 1 #increment the number of losses for this target


# In[9]:


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


# In[10]:


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


# In[11]:


class ThemeManager:
    
    def __init__(self):
        self.primary = (0.5,0.5,0.5,1)        
        self.secondary = (0.9,0.9,0.9,1)
        self.background = (0.0,0.0,0.0,1)
        self.tertiary = (0.6,0.35,0.35,1)
        self.new()
        
    def dark(self, colour):
        return [0.5*val for val in colour[:-1]] + [colour[-1]]        
    def light(self, colour):
        return [1-0.5*(1-val) for val in colour[:-1]] + [colour[-1]]
    def mix(self,c1,c2):
        return [(c1[i]+c2[i])/2 for i in range(4)]
        
        
    def volcano(self):
        self.primary = (0.2,0.15,0.2,1)
        self.secondary = (1,0.1,0.2,1)
        self.background = (0.0,0.0,0.0,1)
        Window.clearcolor = self.background
    def rhubarb(self):
        self.primary = (0.9,0.6,0.6,1)
        self.secondary = (1,1,0.9,1)
        self.background = (0.7,1,0.3,1)
        Window.clearcolor = self.background
    def fresh(self):
        self.primary = (0.6,0.8,0.6,1)
        self.secondary = (1,1,0.9,1)
        self.background = (0.4,0.6,0.7,1)
        Window.clearcolor = self.background
    def steel(self):
        self.primary = (0.2,0.2,0.25,1)        
        self.secondary = (0.9,0.9,1,1)
        self.background = (0,0,0,1)
        Window.clearcolor = self.background
    def moonlit(self):
        self.primary = (0.3,0.2,0.4,1)        
        self.secondary = (0.95,0.9,1,1)
        self.background = (0,0,0,1)
        Window.clearcolor = self.background
    def maths(self):
        self.primary = (0.325,0.3125,0.3,1)        
        self.secondary = (1,1,0.95,1)
        self.background = (0.175,0.175,0.175,1)
        Window.clearcolor = self.background
    def modern(self):
        self.primary = (0.2,0.25,0.3,1)        
        self.secondary = (1,1,1,1)
        self.background = (0.15,0.15,0.15,1)
        Window.clearcolor = self.background
    def mono(self):
        self.primary = (0.2,0.2,0.2,1)        
        self.secondary = (0.7,0.7,0.7,1)
        self.background = (0.1,0.1,0.1,1) 
        Window.clearcolor = self.background
    def new(self):
        self.primary = (0.188,0.180,0.192,1)       
        self.secondary = (0,0.5,0.7,1)#(1,0.3,0.35,1)
        self.background = (0.102,0.094,0.106,1)
        Window.clearcolor = self.background


# In[12]:


#colour = (0.2,0.4,0.8,0.3)
#TM = ThemeManager()
#dark_colour = TM.dark(colour)
#print(dark_colour)
#print(TM.light(colour))


# In[13]:


class MyWindowManager(ScreenManager):
    pass


# In[14]:


class TextLabel(Label):
    pass


# In[15]:


class MultilineTextLabel(Label):
    pass


# In[16]:


class FactorBox(TextLabel):
    pass


# In[17]:


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


# In[18]:


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
        self.ST_graph_points = [(0,0)]
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
                options = list(set(self.training_set)) #create a copy of the training set to use
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
            
            record_win(self.target_number, time_taken, self.game_type)
            
            self.playing = True #start the timer after the first target has been successfully completed
            self.target_animation_stage = 0.0001 #start the animation
            self.ids.target_fx.text = self.ids.target.text
                
            self.ST_graph_points.append((self.game_time, self.score))
            
            self.new_challenge(self.game_time)
            
        self.ids.running_input.text = self.answer
            
    def key_press(self, keyboard, keycode, text, modifiers, other):
        if keycode in keycodes:
            key = keycodes[keycode]
            self.submit(key)
    
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
            #set training set
            def confidence(target_key):
                return data['target_history'][target_key]['wins'] / (1 + data['target_history'][target_key]['losses']**2)
            confidence_values = [confidence(key) for key in data['target_history']]
            confidence_values.sort()
            threshold = confidence_values[int(len(confidence_values)*0.1)]
            self.training_set = []
            for key in data['target_history']:
                conf = confidence(key)
                if conf <= threshold:
                    if key != '0':
                        self.training_set.append(int(key))
            if len(self.training_set) < 2:
                self.training_set = self.training_set + [2,3] #add options to meet minimum requirements
            self.reset()
        else:
            self.game_type = 'normal'
            self.ids.mode_btn.text = 'norm'
            self.ids.mode_btn.color = (0.5,0.5,0.5,1)
            self.reset()
    
    def lose_game(self, losing_attempt='None'):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_over_screen'
        self.manager.screens[1].info_to_display.text = "score: " + str(self.score) + "\ntarget: " + str(self.target_number) + "\nfactors so far: " + str(self.factors_so_far) + "\ninput: " + losing_attempt + "\ncorrect answer: \n" + str(get_factors(self.target_number))
        self.manager.screens[1].update_graphs(self.game_type)
        
    def show_info(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'info_screen'
        
    def show_stats(self):
        self.manager.transition.direction = 'right'
        self.manager.screens[3].update_info()
        self.manager.current = 'stats_screen'
        


# In[19]:


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
        self.ST_plot.points = self.manager.screens[0].ST_graph_points
        
        this_score = self.ST_plot.points[-1][1]
        this_run_time = self.ST_plot.points[-1][0]
        best_run_time = None #set this in the code below
        
        if game_type == 'normal':
            #check against saved high score
            if data['best_run'] == None:
                data['best_run'] = list(self.ST_plot.points)
                high_score = data['best_run'][-1][1]
                best_run_time = data['best_run'][-1][0]
            else:
                high_score = data['best_run'][-1][1]
                best_run_time = data['best_run'][-1][0]
                if this_score > high_score:
                    data['best_run'] = list(self.ST_plot.points)
                elif (this_score == high_score) and (this_run_time < best_run_time):
                    data['best_run'] = list(self.ST_plot.points)
            self.ST_plot_best.points = data['best_run']
        elif game_type == 'training':
            if data['best_run'] == None:
                self.ST_plot_best.points = [(0,0)]
                best_run_time = 0
                high_score = 0
            else:
                self.ST_plot_best.points = data['best_run']
                best_run_time = data['best_run'][-1][0]
                high_score = data['best_run'][-1][1]
            
        self.ids.ST_graph.xmax = max(this_run_time, best_run_time)+1
        self.ids.ST_graph.x_ticks_major = graph_tick_dist(self.ids.ST_graph.xmax)
        self.ids.ST_graph.ymax = max(this_score, high_score)+1
        self.ids.ST_graph.y_ticks_major = graph_tick_dist(self.ids.ST_graph.ymax)

        save(data)
        
    def refresh_scatterplots(self):
        return
        
    def restart(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[20]:


class Info(Screen):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def restart(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()
        
    def search(self, link):
        import webbrowser
        try:
            webbrowser.open(link)
        except ex as Exception:
            print('error opening link')


# In[21]:


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
    
    def update_info(self):
        if not data['best_run'] == None:
            self.best_ST_plot.points = data['best_run']
            self.ids['highscore'].score = str(data['best_run'][-1][1])
            self.ids['highscore'].time = str(int(data['best_run'][-1][0]))
        self.WRN_plot.points = []
        self.TN_plot.points = []

        #iterate through data and add points to graphs as we go
        for key in data['target_history'].keys():
            wins = data['target_history'][key]['wins']
            losses = data['target_history'][key]['losses']
            avg_time = data['target_history'][key]['avg_time']
            if (wins+losses)>3:
                self.WRN_plot.points.append((int(key), wins/(wins+losses)))
            if avg_time != 0:
                self.TN_plot.points.append((int(key), avg_time))
                
        self.refresh_graphs()
    
    def refresh_graphs(self):
                
        if data['best_run']!=None and len(data['best_run'])>0:
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
    
    def restart(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[22]:


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
        Clock.schedule_interval(my_grid.update, 1.0/60.0)
        
        self.SM.add_widget(my_grid)
        self.SM.add_widget(game_over)
        self.SM.add_widget(info)
        self.SM.add_widget(stats)
        #print(dir(self))
                        
        Window.bind(on_key_down = my_grid.key_press)
        
        return self.SM
        
if __name__ == '__main__':
    app = PrimeFactorizerApp()
    app.run()


# In[ ]:




