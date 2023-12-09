# List of modules

## Module __init__.py

## Module: data_extraction

- Class Data
    - Function __init__ : None → none
    - Function extract_from_query : query:str → None
    - Function extract_from_constraints : constraints:dict-like → None
    - Function extract_from_file : data:str → None
    - Function get_spectra : None -> None
    - Function write_file : path:str → None
    - Function concat: new_data:Data, axis:int → None
    - Function merge: new_data:Data, on_column:str →None
    - Attribute : data : ~pd.Dataframe() or equivalent
    - Extract metadata functions (we are already using pandas dataframe so no need to do more ?): 
        - pandas like functions to extract data as regards identifiers, coordinates, chemical abundances, redshifts, or other fields requested by end-user 
## Module: preprocessing 
- Class Preprocessing
    - Function (static) normalize : data → pd.DataFrame
    - Function (static) remove_outliers : data → pd.DataFrame
    - Function (static) interpolate : data → array-like, array-like
    - Function (static) correct_redshift : data → pd.DataFrame

## Module: wavelength_alignment
- Class WavelengthAlignment
    - Function: align : data → None
    - Function: interpolate: data → None

## Module: visualization
- Class Visualization
    - Function (static) plot : data -> plt.figure
    visualize spectra with an overlay of the inferred continuum

## Module: interactive_visualization
- Class InteractiveVisualization
    - Function (static) plot_interactive : data -> None
    enable users to select plot regions and quantify the flux of spectral lines in an interactive mode

## Module: data_augmentation
- Class DataAugmentor
    - Function (static) compute_derivative : Data , modify_original_dataset: bool-> Data or None is  modify_original_dataset=True
        - Compute first order derivatives with respect to wavelength(?)
    - Function (static) compute_frac_derivative : Data , modify_original_dataset: bool-> Data or None is  modify_original_dataset=True
        - Compute fractional derivative of the spectra

## Module: classification 
- Class: Classifier: (based on sklearn classes)
    - Function: __init__ : model_name:str, **kwargs -> None
    - Function: fit : x_train:array-like, y_train:array-like → None
    - Function: predict : x:array-like → array-like
    - Function: predict_proba : x:array-like → array-like 
    - Function: score : x:array-like, y_true :array-like → float
    - Function: confusion_matrix : y_true:array-like, y_pred:array-like -> array-like
