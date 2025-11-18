import os, glob
import sys
import json
import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit as cf
from scipy.signal import savgol_filter
import scipy.integrate as integrate
from PIL import Image
import darkdetect

version_number = "25/11"
Standard_path = os.path.dirname(os.path.abspath(__file__))

Delta_lambd = 0.25e-9
hc = 1.24e-4   # planck constant * speed of light per 1cm in [eV]
kbT = 0.025266 # Energy of room temperature
kb = 8.617333e-5 # Boltzmann constant in eV/K
c = 3e10       # speed of light in cm/s

def set_plot_params():
    plt.rcParams["figure.figsize"] = (8,4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust the value to make it thinner
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Cross Section Spectroscopy."+version_number)
        self.geometry("1280x720")

        set_plot_params()
        self.initialize_variables()
        self.setup_plot_area()
        self.initialize_ui_images()
        self.initialize_ui()

        self.toplevel_window = {'Plot Settings': None,
                                'Legend Settings': None}

        self.load_settings_frame()

        self.canvas_width.bind("<KeyRelease>", lambda val: self.update_canvas_size(self.canvas_ratio_list[self.canvas_ratio.get()]))
        self.canvas_height.bind("<KeyRelease>", lambda val: self.update_canvas_size(self.canvas_ratio_list[self.canvas_ratio.get()]))

        self.save_attributes = ["material_list", "use_McCumber", "use_Fuchtbauer", "average_sigma"]

        self.save_attributes.extend(["doping", "thickness", "tau_f", "refractive_index", "temperature"])
        self.save_attributes.extend(["zero_bandwidth", "FF_absorption", "FF_fluorescence", "savgol_filter", "lower_zero_index", "higher_zero_index"])
        self.save_attributes.extend(["MC_central", "MC_width", "FL_absorption"])
        self.save_attributes.extend(self.settings_widgets)

    def initialize_ui_images(self):
        self.img_settings = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","options.png")), size=(15, 15))
        self.img_save = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","save_white.png")), size=(15, 15))
        self.img_folder = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","folder.png")), size=(15, 15))
        self.img_fluorescence = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","fluorescence.png")), size=(15, 15))
        self.img_absorption = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","absorption.png")), size=(15, 15))

    def initialize_variables(self):
        self.materials = [f for f in os.listdir(os.path.join(Standard_path, "measurements"))]

        self.ax = None
        self.plot_index = 0
        self.color = "#212121" # toolbar
        self.text_color = "white"

        # line plot settings
        self.moving_average = 1
        
        # Boolean variables
        self.canvas_ratio_list = {'Auto': None, 'Custom': 0,'4:3 ratio': 4/3, '16:9 ratio': 16/9, '3:2 ratio': 3/2, '3:1 ratio': 3,'2:1 ratio': 2, '1:1 ratio': 1, '1:2 ratio': 0.5}

    # user interface, gets called when the program starts 
    def initialize_ui(self):
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, height=600, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=999, sticky="nesw")
        self.sidebar_frame.rowconfigure(20, weight=1)
        self.rowconfigure(11,weight=1)

        
        # SIDEBAR FRAME 
        frame = self.sidebar_frame
        App.create_label(frame, row=0, column=0, text="CSS v."+version_number, font=customtkinter.CTkFont(size=20, weight="bold"), padx=20, pady=(10,15), sticky=None)

        #buttons
        self.material_list             = App.create_Menu(frame, values=self.materials, column=0, row=1, command=self.load_material, init_val=self.materials[0])
        self.plot_fluorescence_button  = App.create_button(frame, text="Plot fluorescence", command=self.fluorescence_plot, column=0, row=4, image=self.img_fluorescence, sticky="w")
        self.plot_absorption_button    = App.create_button(frame, text="Plot absorption", command=self.absorption_plot, column=0, row=5, image=self.img_absorption, sticky="w")
        self.plot_cross_section_button = App.create_button(frame, text="Plot cross section", command=self.cross_sections_plot, column=0, row=6, sticky="w")

        # bottom settings
        self.save_button    = App.create_button(frame, text="Save figure/data", command=self.save_figure,     column=0, row=23,  image=self.img_save, pady=(5,15))
        
        #switches
        self.config_title       = App.create_label(frame, row=10, column=0, text="Configure Simulation", font=customtkinter.CTkFont(size=16, weight="bold"), padx=20, pady=(20, 5),sticky=None)
        self.use_McCumber       = App.create_switch(frame, row=11, column=0, text="Use McCumber")
        self.use_Fuchtbauer     = App.create_switch(frame, row=12, column=0, text="Use Füchtbauer", pady=(5,15))
        self.crystal_button     = App.create_switch(frame, row=13, column=0, text="Config Crystal", command=lambda: self.toggle_sidebar_window(self.crystal_button, self.crystal_widgets))
        self.fluorescence_button  = App.create_switch(frame, row=14, column=0, text="Config Fluorescence", command=lambda: self.toggle_sidebar_window(self.fluorescence_button, self.fluorescence_widgets))
        self.absorption_button  = App.create_switch(frame, row=15, column=0, text="Config Absorption", command=lambda: self.toggle_sidebar_window(self.absorption_button, self.absorption_widgets))
        self.McCumber_button    = App.create_switch(frame, row=16, column=0, text="Config McCumber", command=lambda: self.toggle_sidebar_window(self.McCumber_button, self.cross_section_widgets))


        # SETTINGS FRAME
        frame = self.tabview.tab("Settings")
        self.load_button         = App.create_button(frame, row=0, column=0, text="Set Working directory", command=self.read_file_list, columnspan=2, image=self.img_folder, width=250)
        self.folder_path         = App.create_entry(frame, row=0, column=3, text="Folder path", columnspan=2, width=600, pady=10, init_val=Standard_path)
        self.json_path           = App.create_entry(frame, row=1, column=3, text="project file name", columnspan=2, width=600, pady=10, init_val="project_data")
        self.save_data_button    = App.create_button(frame, row=1, column=0, text="save project", command=lambda: self.save_project(os.path.join(self.folder_path.get(), f"{self.json_path.get()}.json")), image=self.img_save, width=110)
        self.load_data_button    = App.create_button(frame, row=1, column=1, text="load project", command=lambda: self.load_project(os.path.join(self.folder_path.get(), f"{self.json_path.get()}.json")), image=self.img_folder, width=110)

        self.plot_settings_title = App.create_label(frame, row=2, column=0, text="Plot settings", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=2, padx=20, pady=(20, 5),sticky=None)
        self.show_title          = App.create_switch(frame, row=3, column=0, text="Show title", pady=(10,5), columnspan=2)
        self.show_grid           = App.create_switch(frame, row=4, column=0, text="Use Grid", command=self.toggle_grid, columnspan=2)

        self.canvas_size_title = App.create_label(frame, row=9, column=0, text="Canvas Size", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=2, padx=20, pady=(20, 5),sticky=None)
        self.canvas_width, self.canvas_width_label        = App.create_entry(frame,column=1, row=11, width=70,text="width in cm", placeholder_text="10 [cm]", sticky='w', init_val=10, textwidget=True)
        self.canvas_height, self.canvas_height_label      = App.create_entry(frame,column=1, row=12, width=70,text="height in cm", placeholder_text="10 [cm]", sticky='w', init_val=10, textwidget=True)
        self.canvas_ratio   = App.create_Menu(frame, column=1, row=10, width=110, values=list(self.canvas_ratio_list.keys()), text="Canvas Size", command=lambda x: self.update_canvas_size(self.canvas_ratio_list[x]))

        self.settings_widgets = ["show_title", "show_grid", "canvas_width", "canvas_height", "canvas_ratio"]

        self.show_title.select()
        self.show_grid.select()
        self.use_McCumber.select()
        self.use_Fuchtbauer.select()

    # initialize all widgets on the settings frame
    def load_settings_frame(self):
        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.settings_frame.grid_columnconfigure(0, minsize=60)
        self.columnconfigure(2,weight=1)

        self.load_crystal_sidebar()
        self.load_fluorescence_sidebar()
        self.load_absorption_sidebar()
        self.load_cross_section_sidebar()
        self.load_material(self.materials[0])


    def create_label(self, row, column, width=20, text=None, anchor='e', sticky='e', textvariable=None, padx=(5,5), image=None, pady=None, columnspan=1, fg_color=None, **kwargs):
        label = customtkinter.CTkLabel(self, text=text, textvariable=textvariable, width=width, image=image, anchor=anchor, fg_color=fg_color,  **kwargs)
        label.grid(row=row, column=column, sticky=sticky, columnspan=columnspan, padx=padx, pady=pady)
        return label

    def create_entry(self, row, column, width=200, height=28, text=None, columnspan=1, rowspan=1, padx=10, pady=5, placeholder_text=None, sticky='w', sticky_label='e', textwidget=False, init_val=None, **kwargs):
        entry = CustomEntry(self, width, height, placeholder_text=placeholder_text, **kwargs)
        entry.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            entry.insert(0,str(init_val))
        if text is not None:
            entry_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
            if textwidget == True:
                return (entry, entry_label)

        return entry

    def create_button(self, command, row, column, text=None, image=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkButton(self, text=text, command=command, width=width, image=image, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button
    
    def create_segmented_button(self, values, command, row, column, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkSegmentedButton(self, values=values, command=command, width=width, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button

    def create_switch(self, row, column, text, command=None, columnspan=1, padx=20, pady=5, sticky='w', **kwargs):
        switch = customtkinter.CTkSwitch(self, text=text, command=command, **kwargs)
        switch.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky)
        return switch
    
    def create_combobox(self, values, column, row, width=200, state='readonly', command=None, text=None, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        combobox = customtkinter.CTkComboBox(self, values=values, command=command, state=state, width=width, **kwargs)
        combobox.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e',pady=pady)
        return combobox
    
    def create_Menu(self, values, column, row, command=None, text=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, textwidget=False, init_val=None, **kwargs):
        optionmenu = customtkinter.CTkOptionMenu(self, values=values, width=width, command=command, **kwargs)
        optionmenu.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            optionmenu.set(init_val)
        if text is not None:
            optionmenu_label = App.create_label(self, text=text, column=column-1, row=row, anchor='e', pady=pady)
            if textwidget == True:
                return (optionmenu, optionmenu_label)

        return optionmenu
    
    def create_slider(self, from_, to, row, column, width=200, sliderValue_width=50, text=None, init_val=None, command=None, showSliderValue=False, columnspan=1, number_of_steps=1000, padx = 10, pady=5, sticky='w', **kwargs):
        """
        Create a CTkSlider with optional text label and optional live value display.

        If showSliderValue=True:
        - The slider width is reduced by SliderValue_width.
        - A numeric label is placed in the same column to the right.
        - The label updates automatically as the slider moves.
        - Returns (slider, label, variable)
        """

        # if slider value display is enabled, bind a variable
        variable = customtkinter.DoubleVar(value=init_val if init_val is not None else from_) if showSliderValue else None

        # --- Create slider ---
        effective_width = width - sliderValue_width if showSliderValue else width
        slider = customtkinter.CTkSlider(self, from_=from_, to=to, width=effective_width, command=command, variable=variable, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        returns = [slider]

        if init_val is not None:
            slider.set(init_val)

        # --- Optional text label on the left ---
        if text is not None:
            slider_label = App.create_label(self, text=text, column=column - 1, row=row, width=20, anchor='e')

        # --- Optional numeric value label on the right ---
        if showSliderValue:
            value_label = customtkinter.CTkLabel(self, text=f"{round(slider.get(), 2)}", width=sliderValue_width, anchor='e')
            value_label.grid(column=column + columnspan-1, row=row, padx=(10+width-sliderValue_width, 10), sticky='e')

            # live update on slider movement
            def update_value_label(*_):
                value_label.configure(text=f"{round(variable.get(), 2)}")
                if command:
                    command(variable.get())

            slider.configure(command=lambda v: variable.set(float(v)))
            variable.trace_add("write", update_value_label)
            returns.append(variable)

        # Unpack if only one element, else return tuple
        return returns[0] if len(returns) == 1 else tuple(returns)
     
    def create_table(self,  width, row, column, sticky=None, rowspan=1, **kwargs):
        text_widget = customtkinter.CTkTextbox(self, width = width, padx=10, pady=5)
        # text_widget.pack(fill="y", expand=True)
        text_widget.grid(row=row, column=column, sticky=sticky, rowspan=rowspan, **kwargs)
        self.grid_rowconfigure(row+rowspan-1, weight=1)

        return text_widget
    
    def create_textbox(self, row, column, width=200, height=28, text=None, columnspan=1, rowspan=1, padx=10, pady=5, sticky='w', sticky_label='e', textwidget=False, init_val=None, **kwargs):
        textbox = customtkinter.CTkTextbox(self, width, height, **kwargs)
        textbox.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            textbox.insert(0,str(init_val))
        if text is not None:
            textbox_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
            if textwidget == True:
                return (textbox, textbox_label)

        return textbox
    
    # Load the sidebar
    def load_crystal_sidebar(self):
        row = 0
        before_widgets = set(self.settings_frame.winfo_children())

        App.create_label(self.settings_frame, row=row, column=0, text="Crystal Settings", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=4, padx=20, pady=(20, 5), sticky=None)
        self.doping        = App.create_entry(self.settings_frame, row=row+1, column=1, width=110, text="doping [cm⁻³]")
        self.thickness     = App.create_entry(self.settings_frame, row=row+2, column=1, width=110, text="thickness [mm]")
        self.tau_f         = App.create_entry(self.settings_frame, row=row+3, column=1, width=110, text="lifetime τ [ms]")
        self.refractive_index = App.create_entry(self.settings_frame, row=row+4, column=1, width=110, text="refractive index")
        self.temperature    = App.create_entry(self.settings_frame, row=row+5, column=1, width=110, text="temperature [K]")

        for widget in [self.doping, self.thickness, self.tau_f, self.refractive_index, self.temperature]:
            widget.bind("<KeyRelease>", lambda val: self.update_material_dictionary(val))

        self.crystal_widgets = set(self.settings_frame.winfo_children()) - before_widgets
        self.toggle_sidebar_window(self.crystal_button, self.crystal_widgets)

    def load_fluorescence_sidebar(self):
        row = 10

        before_widgets = set(self.settings_frame.winfo_children())
        App.create_label(self.settings_frame, text="Fluorescence Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=4, padx=20, pady=(20, 5),sticky=None)
        self.FF_fluorescence, self.FF_fluorescence_var        = App.create_slider(self.settings_frame, from_=0, to=1, column=1, row=row+2, width=150, text="fourier filter (raw data)", init_val=0.6, number_of_steps=100, showSliderValue=True, command= self.update_fluo_slider_value)

        self.fluorescence_widgets = set(self.settings_frame.winfo_children()) - before_widgets
        self.toggle_sidebar_window(self.fluorescence_button, self.fluorescence_widgets, First_time=True)

    def load_absorption_sidebar(self):
        row = 20

        before_widgets = set(self.settings_frame.winfo_children())
        App.create_label(self.settings_frame, text="Absorption Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=4, padx=20, pady=(20, 5),sticky=None)
        self.FF_absorption, self.FF_absorption_var        = App.create_slider(self.settings_frame, from_=0, to=1, column=1, row=row+1, width=150, text="fourier filter (raw data)", init_val=0, number_of_steps=100, showSliderValue=True, command= self.update_abs_slider_value)
        self.savgol_filter, self.savgol_filter_var          = App.create_slider(self.settings_frame, from_=0, to=50, column=1, row=row+2, width=150, text="Savitzky Golay filter (σ)", init_val=0, number_of_steps=50, showSliderValue=True, command= self.update_abs_slider_value)
        self.zero_bandwidth, self.zero_bandwidth_var        = App.create_slider(self.settings_frame, from_=0, to=100, column=1, row=row+3, width=150, text="zero abs. bandwidth [nm]", init_val=0, number_of_steps=100, showSliderValue=True, command= self.update_abs_slider_value)
        self.lower_zero_index, self.lower_zero_index_var    = App.create_slider(self.settings_frame, from_=0, to=1, column=1, row=row+4, width=150, text="zero wavelength 1 [nm]", init_val=1, number_of_steps=100, showSliderValue=True, command= self.update_abs_slider_value)
        self.higher_zero_index, self.higher_zero_index_var  = App.create_slider(self.settings_frame, from_=0, to=1, column=1, row=row+5, width=150, text="zero wavelength 2 [nm]", init_val=1, number_of_steps=100, showSliderValue=True, command= self.update_abs_slider_value)

        self.absorption_widgets = set(self.settings_frame.winfo_children()) - before_widgets
        self.toggle_sidebar_window(self.absorption_button, self.absorption_widgets, First_time=True)

    def load_cross_section_sidebar(self):
        row = 30
        before_widgets = set(self.settings_frame.winfo_children()) # font=customtkinter.CTkFont(size=16, weight="bold")
        self.average_sigma = App.create_switch(self.settings_frame, row=row, column=0, text="Average MC Cumber", columnspan=4, padx=20, pady=(20, 5),sticky=None, font=customtkinter.CTkFont(size=16, weight="bold"), command=lambda: self.cross_sections_plot())
        self.MC_central, self.MC_central_var = App.create_slider(self.settings_frame, from_=0, to=1, column=1, row=row+2, width=150, text="MC central WL", init_val=0, number_of_steps=100, showSliderValue=True, command=lambda value: self.update_cross_sections_plot())
        self.MC_width, self.MC_width_var = App.create_slider(self.settings_frame, from_=0, to=50, column=1, row=row+3, width=150, text="average bandwidth", init_val=0, number_of_steps=100, showSliderValue=True, command=lambda value: self.update_cross_sections_plot())

        self.FL_title = App.create_label(self.settings_frame, row=row+5, column=0, text="FL Settings", font=customtkinter.CTkFont(size=16, weight="bold"), columnspan=4, padx=20, pady=(20, 5),sticky=None)
        self.FL_absorption, self.FL_absorption_var = App.create_slider(self.settings_frame, from_=0, to=3, column=1, row=row+6, width=150, text="absorption depth [mm]", init_val=0, number_of_steps=100, showSliderValue=True, command=lambda value: self.update_cross_sections_plot())

        # for widget in [self.MC_central, self.MC_width, self.FL_absorption]:
        #     widget.bind("<KeyRelease>", lambda val: self.update_material_dictionary(val))

        self.cross_section_widgets = set(self.settings_frame.winfo_children()) - before_widgets
        self.toggle_sidebar_window(self.McCumber_button, self.cross_section_widgets, First_time=True)

    def update_material_dictionary(self, value):
        self.material_dict["N_dop"] = float(self.doping.get())*1e6
        self.material_dict["length"] = float(self.thickness.get())*1e-3
        self.material_dict["tau_f"] = float(self.tau_f.get())*1e-3
        self.material_dict["zero_absorption_width"] = self.zero_bandwidth.get()
        self.material_dict["zero_absorption_wavelength"] = (int(self.lower_zero_index.get()), int(self.higher_zero_index.get()))
        self.material_dict["absorption_depth"] = float(self.FL_absorption.get())
        self.material_dict["temperature"] = float(self.temperature.get())
        self.material_dict["n"] = float(self.refractive_index.get())

    # load the material
    def load_material(self, material):
        path = os.path.join(Standard_path, "measurements", material, "basedata.json")
        with open(path, "r") as file:
            self.material_dict = json.load(file)

        self.material_dict.setdefault("zero_absorption_wavelength", (0, np.inf))
        self.material_dict.setdefault("folder_path", material)
        self.material_dict.setdefault("zero_absorption_width", 0)

        self.doping.reinsert(self.material_dict["N_dop"]*1e-6)
        self.thickness.reinsert(str(self.material_dict["length"]*1e3))
        self.tau_f.reinsert(str(self.material_dict["tau_f"]*1e3))
        self.refractive_index.reinsert(str(self.material_dict["n"]))
        self.temperature.reinsert(str(self.material_dict["temperature"]))
        self.zero_bandwidth.set(self.material_dict["zero_absorption_width"])

        self.E_u = self.material_dict.get("energy_upper_level", [1e-2/self.material_dict["ZPL"]])
        self.E_l = self.material_dict.get("energy_lower_level", [0])

        try:
            sigma_a, absorption, reference, ratio = calc_absorption(self.material_dict, filter_width=float(self.FF_absorption.get()), savgol_filter_width=int(self.savgol_filter.get()))
            self.higher_zero_index.configure(from_=absorption[0,0], to=absorption[-1,0])
            self.lower_zero_index.configure(from_=absorption[0,0], to=absorption[-1,0])
            self.higher_zero_index.set(absorption[-1,0])
            self.lower_zero_index.set(absorption[0,0])

            self.MC_central.configure(from_=absorption[0,0], to=absorption[-1,0])
            

        except:
            FileNotFoundError("Absorption or reference data not found in the selected material folder.")

        self.FF_absorption_var.set(0)
        self.MC_central.set(self.material_dict["ZPL"]*1e9)
        self.MC_width.set(10)
        
    def toggle_sidebar_window(self, button, widgets, First_time=False):
        if button.get():
            self.settings_frame.grid()
            [widget.grid() for widget in widgets]
        else:
            [widget.grid_remove() for widget in widgets]
            self.close_sidebar_window()
        
        if button == self.absorption_button and not First_time:
            self.absorption_plot()

    def toggle_grid(self):
        self.ax.grid(self.show_grid.get())
        plt.rcParams["axes.grid"] = self.show_grid.get()

    def close_sidebar_window(self):
        if not self.crystal_button.get() and not self.absorption_button.get() and not self.McCumber_button.get() and not self.fluorescence_button.get():
            self.settings_frame.grid_remove()

    # Plotting section
    def setup_plot_area(self):
        """Call this ONCE when initializing the GUI"""
        self.tabview = customtkinter.CTkTabview(self, width=250, command=lambda: self.toggle_toolbar(self.tabview.get()))
        self.tabview.grid(row=3, column=1, padx=(10, 10), pady=(20, 0), columnspan=2, sticky="nsew", rowspan=10)
        self.tabview.add("Show Plots")
        self.tabview.add("Settings")
        self.tabview.tab("Show Plots").columnconfigure(0, weight=1)
        self.tabview.tab("Show Plots").rowconfigure(0, weight=1)

        self.fig = plt.figure(constrained_layout=True, dpi=150)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
        self.canvas_widget = self.canvas.get_tk_widget()
        self.toolbar = self.create_toolbar()
        self.canvas_widget.pack(fill="both", expand=True)
        self.canvas.draw()

    def clear_figure(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def fluorescence_plot(self):
        self.clear_figure()

        # get fluorescence file names!
        path = os.path.join(Standard_path, "measurements", self.material_dict["folder_path"])
        fluorescence_files = glob.glob(os.path.join(path, '*fluorescence*.txt')) 
        filenames = [os.path.basename(f).replace(".txt", "").replace("_", " ") for f in fluorescence_files]
    
        if len(fluorescence_files) > 1: 
            self.line_fluo_low, = self.ax.plot([], [], label=filenames[0])
            self.line_fluo_high, = self.ax.plot([], [], label=filenames[1])

        self.line_fluo, = self.ax.plot([], [], label="fluorescence")

        self.ax.set_xlabel("wavelength in nm")
        self.ax.set_ylabel("fluorescence intensity in a.u.")
        self.ax.legend()
        if self.show_title.get(): self.ax.set_title(f"fluorescence of {self.material_dict['name']}")
        self.update_fluorescence_plot()

    def update_fluorescence_plot(self):
        if hasattr(self, 'line_fluo'):
            Fluo, Fluo_low, Fluo_high = calc_fluorescence(self.material_dict, filter_width=self.FF_fluorescence.get())

            if Fluo_low is not None and Fluo_high is not None: 
                self.line_fluo_low.set_data(Fluo_low[:,0], Fluo_low[:,1])
                self.line_fluo_high.set_data(Fluo_high[:,0], Fluo_high[:,1])

            self.line_fluo.set_data(Fluo[:,0], Fluo[:,1])
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()

    def absorption_plot(self):
        self.clear_figure()

        self.line_abs, = self.ax.plot([],[], label="absorption")
        self.line_ref, = self.ax.plot([],[], label="reference")
        self.line_ref_raw, = self.ax.plot([],[], label="reference raw", c="tab:orange", lw=0.8, alpha=0.7)
        self.line_sigma, = self.ax.plot([],[], label="absorption cross section", c="tab:green", lw=0.8)

        self.ax.set_xlabel("wavelength in nm")
        self.ax.set_ylabel("absorption in a.u.")
        self.ax.legend()

        if self.show_title.get(): self.ax.set_title(f"absorption of {self.material_dict['name']}")

        # vlines stored for later updates
        self.vline_low = self.ax.axvline(0, color='red', linestyle='--', lw=0.8)
        self.vline_high = self.ax.axvline(0, color='red', linestyle='--', lw=0.8)
        self.shade_low = self.ax.axvspan(0,0, alpha=0.1, color='red')
        self.shade_high = self.ax.axvspan(0,0, alpha=0.1, color='red')

        for line in [self.line_ref_raw, self.vline_low, self.vline_high, self.shade_low, self.shade_high]:
            line.set_visible(self.absorption_button.get())

        self.update_absorption_plot()

    def update_absorption_plot(self):
        if not hasattr(self, 'line_abs'):
            return  # plot not initialized yet

        sigma_a, absorption, reference, ratio = calc_absorption(self.material_dict, filter_width=float(self.FF_absorption.get()), savgol_filter_width=int(self.savgol_filter.get()))

        # update plot data
        self.line_abs.set_data(absorption[:,0], absorption[:,1])
        self.line_ref.set_data(reference[:,0], reference[:,1])
        self.line_ref_raw.set_data(reference[:,0], reference[:,1]/ratio)
        self.line_sigma.set_data(sigma_a[:,0], sigma_a[:,1]/(np.max(sigma_a[:,1])/np.max(reference[:,1])))

        # update vertical lines
        val_low  = float(self.lower_zero_index.get())
        val_high = float(self.higher_zero_index.get())
        border_left = val_low - self.zero_bandwidth.get()/2
        border_right = val_high + self.zero_bandwidth.get()/2

        self.vline_low.set_xdata([val_low, val_low])
        self.vline_high.set_xdata([val_high, val_high])
        self.shade_low.set_x(max(val_low - self.zero_bandwidth.get()/2, absorption[0,0]))
        self.shade_low.set_width(self.zero_bandwidth.get()-(max(border_left, absorption[0,0]) - border_left))
        self.shade_high.set_x(val_high - self.zero_bandwidth.get()/2)
        self.shade_high.set_width(self.zero_bandwidth.get()+(min(border_right, absorption[-1,0]) - border_right))

        # refresh only the artists (faster than full draw)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def cross_sections_plot(self):
        self.clear_figure()
        # absorption_depth in cm, accounts for reabsorption in the crystal

        self.sigma_a = calc_absorption(self.material_dict, filter_width=float(self.FF_absorption.get()), savgol_filter_width=int(self.savgol_filter.get()))[0]

        plot_list = [self.sigma_a]
        plot_list_labels = [f"$\\sigma_a$ {self.material_dict['name']}"]
        plot_list_names = ["line_sigma_a", "line_sigma_e"]

        if self.use_Fuchtbauer.get():
            Fluo = calc_fluorescence(self.material_dict)[0]
            self.sigma_e = Fuchtbauer_Ladenburg(Fluo, self.material_dict, sigma_a=self.sigma_a, absorption_depth=self.material_dict["absorption_depth"])
            plot_list += [self.sigma_e]
            plot_list_labels += [f"$\\sigma_e$ Füchtbauer"]

        self.ax.set_xlabel("wavelength in nm")
        self.ax.set_ylabel("cross sections in cm²")
        if self.show_title.get(): self.ax.set_title(f"cross sections of {self.material_dict['name']}")

        if self.use_McCumber.get():
            thermal_energy = kb * self.material_dict.get("temperature", 295)  # in eV

            self.sigma_e_McCumber = McCumber_relation(self.E_l, self.E_u, self.sigma_a, thermal_energy)
            plot_list += [self.sigma_e_McCumber]
            plot_list_labels += ["$\\sigma_e$ McCumber"]
            plot_list_names += ["line_sigma_e_McCumber"]
            self.ax.set_ylim(-1e-21,2*np.max(self.sigma_a[:,1]))

            if self.use_Fuchtbauer.get() and self.average_sigma.get():
                self.McCumber_line = self.ax.axvline(self.MC_central.get(), color='red', linestyle='--', lw=0.8)
                self.sigma_e_average = average_MCcumber_FL(self.material_dict, self.sigma_e, self.sigma_e_McCumber, self.MC_central.get() - self.MC_width.get()/2, self.MC_central.get() + self.MC_width.get()/2)
                self.sigma_a_average = McCumber_relation(self.E_l, self.E_u, self.sigma_e_average, thermal_energy, inverse_relation=True)
                plot_list += [self.sigma_e_average, self.sigma_a_average]
                plot_list_labels += ["$\\sigma_e$ average", "$\\sigma_a$ average"]
                plot_list_names += ["line_sigma_e_average", "line_sigma_a_average"]
            
            if self.use_Fuchtbauer.get():
                self.ax.set_ylim(-1e-21,1.3*max(np.max(self.sigma_a[:,1]), np.max(self.sigma_e[:,1])))


        for data, label, name in zip(plot_list, plot_list_labels, plot_list_names):
            setattr(self, name, self.ax.plot(data[:,0], data[:,1], label=label)[0])

        self.ax.legend()
        self.canvas.draw()
    
    def update_cross_sections_plot(self):
        if not hasattr(self, 'line_sigma_a'):
            return  # plot not initialized yet

        self.update_material_dictionary(None)
        if self.use_Fuchtbauer.get():
            Fluo = calc_fluorescence(self.material_dict)[0]
            self.sigma_e = Fuchtbauer_Ladenburg(Fluo, self.material_dict, sigma_a=self.sigma_a, absorption_depth=self.material_dict["absorption_depth"])
            self.line_sigma_e.set_data(self.sigma_e[:,0], self.sigma_e[:,1])

        if self.use_McCumber.get() and self.use_Fuchtbauer.get() and self.average_sigma.get():
            thermal_energy = kb * self.material_dict.get("temperature", 295)  # in eV
            # update plot data
            self.sigma_e_average = average_MCcumber_FL(self.material_dict, self.sigma_e, self.sigma_e_McCumber, self.MC_central.get() - self.MC_width.get()/2, self.MC_central.get() + self.MC_width.get()/2)
            self.sigma_a_average = McCumber_relation(self.E_l, self.E_u, self.sigma_e_average, thermal_energy, inverse_relation=True)
            self.line_sigma_e_average.set_data(self.sigma_e_average[:,0], self.sigma_e_average[:,1])
            self.line_sigma_a_average.set_data(self.sigma_a_average[:,0], self.sigma_a_average[:,1])

            # update vertical lines
            val  = float(self.MC_central.get())

            self.McCumber_line.set_xdata([val, val])

        # refresh only the artists (faster than full draw)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
 
    def read_file_list(self):
        path = customtkinter.filedialog.askdirectory(initialdir=self.folder_path)
        if path != "":
            self.folder_path.reinsert(path)

    def create_toolbar(self) -> customtkinter.CTkFrame:
        # toolbar_frame = customtkinter.CTkFrame(master=self.tabview.tab("Show Plots"))
        # toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar_frame = customtkinter.CTkFrame(self)
        toolbar_frame.grid(row=20, column=1, columnspan=2, padx=(10), sticky="ew")
        toolbar = CustomToolbar(self.canvas, toolbar_frame)
        toolbar.config(background=self.color)
        toolbar._message_label.config(background=self.color, foreground=self.text_color, font=(15))
        toolbar.winfo_children()[-2].config(background=self.color)
        toolbar.update()
        return toolbar_frame

        # Closing the application       
    
    def toggle_toolbar(self, value):
        if value == "Show Plots":
            self.toolbar.grid()
        else:
            self.toolbar.grid_remove()
    
    # Save the current figure or the data based on the file type
    def save_figure(self):
        file_name = customtkinter.filedialog.asksaveasfilename()
        if file_name.endswith((".pdf",".png",".jpg",".jpeg",".PNG",".JPG",".svg")): 
            self.fig.savefig(file_name, bbox_inches='tight')
        elif file_name.endswith((".dat",".txt",".csv")):
            all_data = []
            headers = []

            # Collect data
            for i, ax in enumerate(self.fig.axes):
                for j, line in enumerate(ax.get_lines()):
                    x = line.get_xdata()
                    y = line.get_ydata()
                    # make sure lengths match if different lines differ
                    length = min(len(x), len(y))
                    all_data.append(np.column_stack([x[:length], y[:length]]))
                    headers.extend([f"X{i}_{j}", f"Y{i}_{j}"])

            # Align all datasets by rows (pad with blanks if needed)
            max_len = max(arr.shape[0] for arr in all_data)
            aligned = np.full((max_len, len(all_data)*2), "", dtype=object)

            for k, arr in enumerate(all_data):
                aligned[:arr.shape[0], 2*k:2*k+2] = arr

            # Write to file
            with open(file_name, "w") as f:
                f.write("\t".join(headers) + "\n")
                for row in aligned:
                    f.write("\t".join(f"{float(val):.5e}" if val != "" else "" for val in row) + "\n")

    def save_project(self, filename):
        # collect all data-variables to be saved into a dictionary
        project_data = {}
        for name in self.save_attributes:
            val = getattr(self, name)
            # unwrap Tkinter variables automatically
            if isinstance(val, (
                customtkinter.CTkLabel,
                customtkinter.CTkButton,
                customtkinter.CTkSegmentedButton,
                customtkinter.CTkFrame,
                customtkinter.CTkTextbox
            )):
                continue
            if hasattr(val, "get"):
                value = val.get()
                project_data[name] = value

        # Read out the variables and convert to JSON-safe dict
        def to_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()      # convert numpy arrays to lists
            if isinstance(obj, (np.int64, np.float64)):
                return obj.item()        # convert numpy scalars to Python
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_json_safe(v) for v in obj]
            return obj                   # base case

        safe_data = to_json_safe(project_data)

        with open(filename, "w") as f:
            json.dump(safe_data, f, indent=2)

    def load_project(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        for name, val in data.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                if hasattr(attr, "set"):  # Tkinter variable
                    attr.set(val)
                elif hasattr(attr, "reinsert"):
                    attr.reinsert(val)
                elif hasattr(attr, "select") and val == 1:
                    attr.select()
                elif hasattr(attr, "deselect") and val == 0:
                    attr.deselect()
        
        self.close_sidebar_window()
        self.material_dict["folder_path"] = self.material_list.get()

    def update_abs_slider_value(self, value):
        self.update_material_dictionary(value)
        self.update_absorption_plot()
    
    def update_fluo_slider_value(self, value):
        self.update_material_dictionary(value)
        self.update_fluorescence_plot()

    def update_canvas_size(self, canvas_ratio):
        canvas_width = float(self.canvas_width.get())
        canvas_height = float(self.canvas_height.get())
        if canvas_width <= 2 or canvas_height <= 2: return
        self.canvas_widget.pack_forget()

        self.canvas_height.grid() if canvas_ratio == 0 else self.canvas_height.grid_remove()
        self.canvas_height_label.grid() if canvas_ratio == 0 else self.canvas_height_label.grid_remove()

        if canvas_ratio is not None:
            width = canvas_width/2.54
            height = canvas_height/2.54 if canvas_ratio == 0 else canvas_width/(canvas_ratio*2.54)
            self.fig.set_size_inches(width, height)
            self.canvas_widget.pack(expand=True, fill=None) 
            self.canvas_widget.config(width=self.fig.get_size_inches()[0] * self.fig.dpi, height=self.fig.get_size_inches()[1] * self.fig.dpi)
        else:
            width = (self.tabview.winfo_width() - 18) / self.fig.dpi
            height = (self.tabview.winfo_height()) / self.fig.dpi
            self.fig.set_size_inches(w=width, h=height, forward=True)  # Re-enable dynamic resizing
            self.canvas_widget.pack(fill="both", expand=True) 

        self.canvas.draw()  # Redraw canvas to apply the automatic size

    def on_closing(self):
        try:
            if hasattr(self, "canvas"): self.canvas.get_tk_widget().destroy()
            if hasattr(self, "fig"): plt.close(self.fig)
        except:
            pass
        self.quit()    # Python 3.12 works
        self.destroy() # needed for built exe

##########################################################################
##########################################################################
##########################################################################

class CustomToolbar(NavigationToolbar2Tk):
    # Modify the toolitems list to remove specific buttons
    def set_message(self, s):
        formatted_message = s.replace("\n", ", ").strip()
        self.message.set(formatted_message)

    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        # Add or remove toolitems as needed
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        # ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

# Extend the class of the entry widget to add the reinsert method
class CustomEntry(customtkinter.CTkEntry):
    def reinsert(self, text):
        self.delete(0, 'end')  # Delete the current text
        self.insert(0, text)  # Insert the new text

def linear(x,a,b):
    return -a*x+b

def moving_average(x, window_size):
    # Ensure the window_size is even
    if window_size % 2 == 0:
        half_window = window_size // 2
    else:
        half_window = (window_size - 1) // 2

    if window_size <= 1:
        return x

    half_window = window_size // 2
    cumsum = np.cumsum(x)

    # Calculate the sum of elements for each centered window
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    centered_sums = cumsum[window_size - 1:-1]

    # Divide each sum by the window size to get the centered moving average
    smoothed_array = centered_sums / window_size

    # Pad the beginning and end of the smoothed array with the first and last values of x
    first_value = np.repeat(x[0], half_window)
    last_value = np.repeat(x[-1], half_window)
    smoothed_array = np.concatenate((first_value, smoothed_array, last_value))

    return smoothed_array

def fourier_filter(data, filter_width, Do_plots = False):
    
    if filter_width == 0:
        return data
    
    fft = np.fft.fft(data[:,1])
    fft_filter = np.ones_like(data[:,1])
    mid_index = int(len(fft_filter)/2)
    fft_filter[mid_index-int(filter_width*mid_index):mid_index+int(filter_width*mid_index)] = 0
    
    fft_neu = fft*fft_filter
    filtered_data = np.fft.ifft(fft_neu).real
    
    if Do_plots:
        fig = plt.figure(figsize=(6,4.5),constrained_layout=True, dpi=150) 
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ax1.plot(np.abs(fft))
        ax1.plot(np.abs(fft_neu))
        ax1.set_ylim(0, 0.01*np.max(fft))
        
        ax2.plot(data[:,0], data[:,1])
        ax2.plot(data[:,0], filtered_data)
    
    return np.vstack([data[:,0], filtered_data]).T

def normalize(array):
    return array / (np.sum(array))

def calc_fluorescence(material, filter_width=0.6):
    path = os.path.join(Standard_path, "measurements", material["folder_path"])
    Fluo_low = None
    Fluo_high = None
    
    fluorescence_files = glob.glob(os.path.join(path, '*fluorescence*.txt'))
    
    if len(fluorescence_files) == 1:
        Fluo = np.genfromtxt(fluorescence_files[0], skip_header=2, delimiter=",")

    else: 
        fluo_low_name = fluorescence_files[0]
        fluo_high_name = fluorescence_files[1]
        Fluo_low = np.genfromtxt(fluo_low_name, skip_header=2, delimiter=",")
        Fluo_high = np.genfromtxt(fluo_high_name, skip_header=2, delimiter=",")

        Fluo_low[:,1] = normalize(Fluo_low[:,1])
        Fluo_high[:,1] = normalize(Fluo_high[:,1])
        Fluo = Fluo_low.copy()

        for i, (low, high) in enumerate(zip(Fluo_low[:,1], Fluo_high[:,1])):  
            if abs(high - low) > 1e-5: 
                Fluo[i,1] = min(low,high)
    
    average_interval = find_interval(Fluo[:,0], 990, 1150)
    average_interval2 = find_interval(Fluo[:,0], 1000, 1060)
    Fluo[average_interval,1] = moving_average(Fluo[average_interval,1], 4)
    Fluo[average_interval2,1] = moving_average(Fluo[average_interval2,1], 6)
    Fluo[:,1] = normalize(Fluo[:,1])
    Fluo = fourier_filter(Fluo, filter_width = filter_width)

    return Fluo, Fluo_low, Fluo_high

def calc_cubic_interpolation(absorption, reference, zero_absorption_width, mid_lambda1=0, mid_lambda2=np.inf):
    """
    Perform cubic interpolation between two spectral regions centered around mid_idx1 and mid_idx2.

    Parameters
    ----------
    absorption : np.ndarray
        2D array with columns [x, y_absorption].
    reference : np.ndarray
        2D array with columns [x, y_reference].
    zero_absorption_width : float
        Width in the same units as absorption[:,0] around each center index used for interpolation regions.
    mid_lambda1 : float, optional
        Center wavelength for the first region (default 0, start of array).
    mid_lambda2 : float, optional
        Center wavelength for the second region (default np.inf, end of array).

    Returns
    -------
    np.ndarray
        Interpolated values of the cubic polynomial evaluated over absorption[:,0].
    """
    dlambda = absorption[1,0] - absorption[0,0]
    w = int(zero_absorption_width / dlambda)  # Convert width in nm to number of pixels
    n = len(absorption)
    mid_idx1 = np.argmin(np.abs(absorption[:,0] - mid_lambda1))
    mid_idx2 = np.argmin(np.abs(absorption[:,0] - mid_lambda2)) if mid_lambda2 != np.inf else len(absorption) - 1

    # Handle default special case: only use end section if w == 0
    if w == 0:
        y1 = absorption[mid_idx1, 1]/reference[mid_idx1, 1]
        y2 = absorption[mid_idx2, 1]/reference[mid_idx2, 1]

        if mid_idx1 == mid_idx2: return y1  # both indices are the same
        
        x = np.arange(n)
        return y1 + (x-mid_idx1) * (y2 - y1) / (mid_idx2 - mid_idx1)
        # return np.mean(absorption[-10:,1]) / np.mean(reference[-10:,1])

    # Define start and end slices for both regions
    region1 = slice(max(0, mid_idx1 - w//2), min(n, mid_idx1 + w//2))
    region2 = slice(max(0, mid_idx2 - w//2), min(n, mid_idx2 + w//2))

    # Subdivide each region into two averaged sections (for total of 4 interpolation points)
    def region_points(region):
        idx = np.arange(region.start, region.stop)
        sublen = max(1, len(idx) // 5)
        y1 = np.mean(absorption[idx[:sublen],1]) / np.mean(reference[idx[:sublen],1])
        y2 = np.mean(absorption[idx[-sublen:],1]) / np.mean(reference[idx[-sublen:],1])
        x1 = np.mean(absorption[idx[:sublen],0])
        x2 = np.mean(absorption[idx[-sublen:],0])
        return [(x1, y1), (x2, y2)]

    points = region_points(region1) + region_points(region2)
    x_values, y_values = np.array(points).T

    # Solve cubic polynomial
    A = np.vander(x_values, 4)
    coefficients = np.linalg.solve(A, y_values)
    poly = np.polynomial.Polynomial(coefficients[::-1])

    return poly(absorption[:,0])

def calc_absorption(material, filter_width = 0, savgol_filter_width = 20, savgol_filter_order=3):
    path = os.path.join(Standard_path, "measurements", material["folder_path"])

    absorption_files = [f for f in glob.glob(os.path.join(path, '*absorption*.txt')) if 'reference' not in f.lower()]
    reference_files = glob.glob(os.path.join(path, '*reference*.txt'))

    load_txt = lambda f: np.genfromtxt(f, skip_header=2, delimiter=",")

    absorption_spectra = [load_txt(f) for f in absorption_files]
    reference_spectra = [load_txt(f) for f in reference_files]

    absorption = join_spectra(absorption_spectra) if len(absorption_spectra) > 1 else absorption_spectra[0]
    reference  = join_spectra(reference_spectra)  if len(reference_spectra)  > 1 else reference_spectra[0]

    # Assuming: absorption[:,0] and reference[:,0] are x-values
    x_min = max(absorption[:,0].min(), reference[:,0].min())
    x_max = min(absorption[:,0].max(), reference[:,0].max())

    # Trim to overlapping region
    absorption = absorption[(absorption[:,0] >= x_min) & (absorption[:,0] <= x_max)]
    reference = reference[(reference[:,0] >= x_min) & (reference[:,0] <= x_max)]

    if absorption.shape[0] != reference.shape[0]:
        # Interpolate to common x-values
        reference_interp = np.interp(absorption[:,0], reference[:,0], reference[:,1])
        reference = np.vstack([absorption[:,0], reference_interp]).T

    reference = fourier_filter(reference, filter_width = filter_width)
    absorption = fourier_filter(absorption, filter_width = filter_width)
    
    mid_wavelength = material.get("zero_absorption_wavelength")
    # ratio = np.mean(absorption[-20:,1]) / np.mean(reference[-20:,1])
    ratio = calc_cubic_interpolation(absorption, reference, material['zero_absorption_width'], mid_lambda1=mid_wavelength[0], mid_lambda2=mid_wavelength[1])
    
    # print(ratio)
    reference[:,1] *= ratio
    
    # Calculate the Absorption 
    sigma_a = np.abs(np.log(reference[:,1]/absorption[:,1]))/(material["N_dop"]*1e-6*material["length"]*1e2)

    if savgol_filter_width > savgol_filter_order:
        sigma_a = savgol_filter(sigma_a, savgol_filter_width, savgol_filter_order)
    
    return np.vstack([absorption[:,0], sigma_a]).T, absorption, reference, ratio

def join_spectra(spectra_list):
    # Join multiple spectra into one, removing overlapping regions by averaging
    if len(spectra_list) == 0:
        return np.array([])

    combined_spectrum = spectra_list[0]

    for next_spectrum in spectra_list[1:]:
        # Find overlapping region
        overlap_start = max(combined_spectrum[0,0], next_spectrum[0,0])
        overlap_end = min(combined_spectrum[-1,0], next_spectrum[-1,0])

        if overlap_start < overlap_end:
            # Indices for overlapping region
            combined_indices = np.where((combined_spectrum[:,0] >= overlap_start) & (combined_spectrum[:,0] <= overlap_end))[0]
            next_indices = np.where((next_spectrum[:,0] >= overlap_start) & (next_spectrum[:,0] <= overlap_end))[0]

            # Average overlapping region
            # x and y values from both spectra in the overlap region
            x1, y1 = combined_spectrum[combined_indices, 0], combined_spectrum[combined_indices, 1]
            x2, y2 = next_spectrum[next_indices, 0], next_spectrum[next_indices, 1]

            # interpolate y2 onto x1 grid
            y2_interp = np.interp(x1, x2, y2)

            # average the y-values on the same x-grid
            averaged_overlap = np.column_stack([x1, (y1 + y2_interp) / 2])

            # Non-overlapping parts
            combined_non_overlap = combined_spectrum[combined_spectrum[:,0] < overlap_start]
            next_non_overlap = next_spectrum[next_spectrum[:,0] > overlap_end]

            # Combine all parts
            combined_spectrum = np.vstack([combined_non_overlap, averaged_overlap, next_non_overlap])
        else:
            # No overlap, just concatenate
            combined_spectrum = np.vstack([combined_spectrum, next_spectrum])

    return combined_spectrum

def calc_partition_function(degeneracies, energies, kbT):
    if not hasattr(degeneracies, "__len__"):
        degeneracies = [degeneracies]*len(energies)
    
    Z = 0 
    for (d, energy) in zip(degeneracies, energies): 
        Z += d*np.exp(-energy / kbT)
    return Z

def calc_Z_lower_upper(energies_lower, energies_upper, kbT):
    # convert energies from cm^-1 to eV
    energies_lower = np.array(energies_lower)*hc
    energies_upper = np.array(energies_upper)*hc

    ZPL = energies_upper[0] - energies_lower[0]
    # print(f"ZPL = {ZPL:.2f}eV = {hc/ZPL*1e7:.2f}nm")
    
    # energy must be measured from the lowest sublevel
    Energy_F72 = energies_lower - energies_lower[0]
    Energy_F52 = energies_upper - energies_upper[0]

    # Calculate the partition functions
    Z_lower = calc_partition_function(2, Energy_F72, kbT)
    Z_upper = calc_partition_function(2, Energy_F52, kbT)

    # print(f"Z_lower = {Z_lower:.3f}\nZ_upper = {Z_upper:.3f}\nZ_l/Z_u = {Z_lower/Z_upper:.3f}")

    return Z_lower, Z_upper, ZPL

def McCumber_relation(energies_lower, energies_upper, sigma_a, kbT, inverse_relation = False):
    # Calculate the emission cross section with the McCumber relation for a given absorption spectrum and the energy levels 
    # if inverse_relation == True, the absorption cross section is calculated from the emission cross section
    sign = -1 if inverse_relation else 1
    # lambdas should be given in cm
    lambdas = sigma_a[:,0]*1e-7   # units: cm
    Z_lower, Z_upper, ZPL = calc_Z_lower_upper(energies_lower, energies_upper, kbT)

    # Calculate the emission cross section
    sigma_e = (Z_lower/Z_upper)**sign * np.exp(sign*(ZPL-hc/lambdas)/kbT) * sigma_a[:,1]
    
    return np.vstack([sigma_a[:,0], sigma_e]).T

def beta_eq(sigma_a, sigma_e):
    return sigma_a / (sigma_a + sigma_e)

def Fuchtbauer_Ladenburg(flourescence, material, sigma_a = None, absorption_depth=0):
    n = material["n"]
    tau = material["tau_f"]
    N_dop = material["N_dop"]*1e-6  # in cm^-3
    lambdas = flourescence[:,0]*1e-7   # units: cm
    # Calculate the emission cross section with the Füchtbauer-Ladenburg relation for a given fluorescence spectrum (wavelengths given in nm)
    
    absorption_cross_section = np.interp(flourescence[:,0], sigma_a[:,0], sigma_a[:,1], left=0, right=0) if sigma_a is not None else np.zeros_like(lambdas)
    # correct for absorption effects, c.f. Toepfer, Jena, 2001, page 43
    absorption_factor = np.exp(N_dop*absorption_cross_section*absorption_depth*0.1)

    Intensity = flourescence[:,1]
    Integral = integrate.simpson(Intensity*lambdas*absorption_factor,x=lambdas)
    g = lambdas**3/c * Intensity * absorption_factor / Integral

    sigma_e = lambdas**2 / (8*np.pi*n**2*tau) * g 
    
    return np.vstack([lambdas*1e7, sigma_e]).T

def find_interval(lambdas, lmin, lmax):
    index_min = np.argmin(np.abs(lambdas-lmin))
    index_max = np.argmin(np.abs(lambdas-lmax))
    return slice(index_min, index_max)

def get_overlap_lengths(arr):
    """Finds the lengths of contiguous patches of ones in a binary array."""
    # Find where the patches start and end
    diff = np.diff(np.concatenate(([0], arr, [0])))  # Add padding to detect edges
    start_indices = np.where(diff == 1)[0]  # Where a 1 starts
    end_indices = np.where(diff == -1)[0]  # Where a 1 ends

    # Compute lengths of patches
    lengths = end_indices - start_indices

    return start_indices, lengths

def average_MCcumber_FL(material, FL_array, MC_array, FL_min=None, MC_max=None):
    Delta_lambd = min(np.diff(FL_array[:,0]).min(), np.diff(MC_array[:,0]).min())
    lambdas = np.arange(min(FL_array[0,0], MC_array[0,0]), max(FL_array[-1,0], MC_array[-1,0]), Delta_lambd)

    FL_array = np.vstack([lambdas, np.interp(lambdas, FL_array[:,0], FL_array[:,1], left=FL_array[0,1], right=FL_array[-1,1])]).T
    MC_array = np.vstack([lambdas, np.interp(lambdas, MC_array[:,0], MC_array[:,1], left=MC_array[0,1], right=MC_array[-1,1])]).T

    array_FL, array_MC = [np.zeros_like(lambdas) for _ in range(2)]
    if FL_min is None: FL_min = material.get("ZPL", 980e-9)*1e9 - 10 
    if MC_max is None: MC_max = material.get("ZPL", 980e-9)*1e9 + 10

    print(FL_min, MC_max)

    sliceFL = find_interval(lambdas, FL_min, 10000)
    sliceMC = find_interval(lambdas, 100, MC_max)

    array_FL[sliceFL] += FL_array[sliceFL,1]
    array_MC[sliceMC] += MC_array[sliceMC,1]

    arrays = [array_FL, array_MC]
    nonzero_mask = np.vstack(arrays) != 0

    # Count nonzero values
    count_nonzero = np.sum(nonzero_mask, axis=0)
    mask_nonzero = count_nonzero - 1
    mask_nonzero[mask_nonzero < 0] = 0
    start_indices, overlap_lengths = get_overlap_lengths(mask_nonzero)

    for i, length in zip(start_indices, overlap_lengths):
        weight = 0.5 * (1+ np.cos(np.linspace(0,np.pi, length)))
        # mask_left[i:i+length] = weight
        # mask_right[i:i+length] = (1-weight)

        for array in arrays:
            if array[i+int(length/2)] > 0:
                if array[i-1] > 0:
                    array[i:i+length] *= weight 
                elif array[i+length+1] > 0:
                    array[i:i+length] *= (1-weight) 

    stacked = np.vstack(arrays)
    average = np.sum(stacked * nonzero_mask, axis=0)

    return np.vstack([lambdas, average]).T

if __name__ == "__main__":

    app = App()
    app.state('normal')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()