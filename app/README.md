Module documentation is available [here](../docs/_build/html/index.html)

# List of modules

## Module: data_extraction

- Class Data
    - Function __init__ : None → none
    - Function extract_from_query : query:str → None
    - Function extract_from_constraints : constraints:dict-like → None
    - Function extract_from_file : path:str → None
    - Function get_spectra_from_obj_id : id:str → list 
    - Function get_spectra_from_data : None → list 
    - Function write_file : path:str → None
    - Function concat: new_data:Data, axis:int → None
    - Function merge: new_data:Data, on_column:str →None
    - Instance Attribute : data:pd.Dataframe
    - Instance Attribute : spectrum:list

## Module: wavelength_alignment
- Class WavelengthAlignment
    - Function align : object_ids:list, min_val:(int,float), max_val:(int,float), num_points:int → pd.DataFrame

## Module: preprocessing 
- Class Preprocessing
    - Function (static) normalize : data:pd.DataFrame → pd.DataFrame
    - Function (static) remove_outliers : data:pd.DataFrame → pd.DataFrame
    - Function (static) interpolate : x:list, y:list, x_lim:Tuple[float, float], bins:int → array-like, array-like
    - Function (static) correct_redshift : data:pd.DataFrame → pd.DataFrame

## Module: visualization
- Class Visualization
    - Function (static) plot : data:pd.DataFrame, y_column:str, order:int=3, figax:Tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]=None, **kwargs -> matplotlib.figure.Figure

## Module: interactive_visualization
- Class InteractiveVisualization
    - Function (static) calc_line_area : x:list[float], y:list[float] -> float
    - Function (static) plot : data:pd.DataFrame, y_column:str, order:int=3, figax:Tuple[matplotlib.figure.Figure,matplotlib.axes.Axes]=None, **kwargs -> matplotlib.widgets.SpanSelector

## Module: data_augmentation
- Class DataAugmentor
    - Function (static) compute_derivative : data:pd.DataFrame, column:str, derivative_order:list[float] -> pd.DataFrame

## Module: classification 
- Class: Classifier:
    - Function __init__ : model_name:str, **kwargs -> None
    - Function fit : x_train:array-like, y_train:array-like → None
    - Function predict : x:array-like → pred:array-like
    - Function predict_proba : x:array-like → pred:array-like 
    - Function score : x:array-like, y_true:array-like → score:float
    - Function confusion_matrix : y_true:array-like, y_pred:array-like -> array-like
    - Function data_for_classifier : data:pd.DataFrame, merge_data:array-like -> pd.DataFrame, array-like
    - Class Attribute: MODEL