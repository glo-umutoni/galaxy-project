# List of modules

## Module __init__.py

## Module: data_extraction

- Class Data
    - Function __init__ : None → none
    - Function extract_from_query : query:str → None
    - Function extract_from_constraints : constraints:dict-like → None
    - Function extract_from_file : data:str → None
    - Function get_spectra_from_obj_id : object_id:str → list (corresponds to output of SDSS.get_spectra())
    - Function get_spectra_from_data : None → list (corresponds to output of SDSS.get_spectra())
    - Function write_file : path:str → None
    - Function concat: new_data:Data, axis:int → None
    - Function merge: new_data:Data, on_column:str →None
    - Attribute : data : pd.Dataframe
    - Attribute : spectrum : list

## Module: preprocessing 
- Class Preprocessing
    - Function (static) normalize : data:pd.DataFrame → pd.DataFrame
    - Function (static) remove_outliers : data:pd.DataFrame → pd.DataFrame
    - Function (static) interpolate : x:list, y:list, x_lim:Tuple[float, float], bins:int → array-like, array-like
    - Function (static) correct_redshift : data:pd.DataFrame → pd.DataFrame

## Module: wavelength_alignment
- Class WavelengthAlignment
    - Function: align : object_ids:list, min_val:(int,float), max_val:(int,float), num_points:int → array-like, list

## Module: visualization
- Class Visualization
    - Function (static) plot : data, y_column:str, order:int=2, figax:Tuple[matplotlib.figure.Figure ,matplotlib.axes.Axes] =None, **kwargs -> matplotlib.figure.Figure
    visualize spectra with an overlay of the inferred continuum

## Module: interactive_visualization
- Class InteractiveVisualization
    - Function (static) plot_interactive : spectra:pd.DataFrame -> plt.figure
    enable users to select plot regions and quantify the flux of spectral lines in an interactive mode

## Module: data_augmentation
- Class DataAugmentor
    - Function (static) compute_derivative : Data , order:float, modify_original_dataset:bool -> Data or None if modify_original_dataset=True
        - Compute first order derivatives with respect to wavelength(?)
    - Function (static) compute_frac_derivative : Data , order:float, modify_original_dataset:bool -> Data or None if modify_original_dataset=True
        - Compute fractional derivative of the spectra

## Module: classification 
- Class: Classifier: (based on sklearn classes)
    - Function: __init__ : model_name:str, **kwargs -> None
    - Function: fit : x_train:array-like, y_train:array-like → None
    - Function: predict : x:array-like → array-like
    - Function: predict_proba : x:array-like → array-like 
    - Function: score : x:array-like, y_true :array-like → float
    - Function: confusion_matrix : y_true:array-like, y_pred:array-like -> array-like
