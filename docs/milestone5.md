## Changes to the API

- **preprocessing module**: We modified the return values of the static functions within 
preprocessing.  Initially the functions normalize, remove_outliers, and correct_redshift
modified the Data class object in place.  However, the SRS clarifications indicate that
each function should be applied to one spectrum, and this alteration makes these
functions more flexible for the user (i.e. they can remove outliers for one spectrum,
but not others).
- **data_extraction module**: We implement a get_spectra_from_obj_id and a get_spectra_from_data methods that obtains the spectra 
for the sky objects represented in the Data class or using a given object id. This functionality simplifies spectra extraction for the end user. We also added a spectrum attribute that makes integration easier, since the spectra data can be extracted once, stored with the object instance, and passed directly to several other modules that require it. 
- **wavelength_alignment module**: We removed the interpolate method, as it was redundant with the preprocessing module. TO stick to the contract, we decided  to group interpolation together with the other preprocessing functions, and have the wavelength_alignment module call it. We also changed the function signature of the align method to work with spectra data and a user-specified range of target wavelengths, as specified in the SRS clarification.
- **visualization module**: We changed the function signature of the plot method to take in a spectrum in the form of a pandas DataFrame, which integrates with the new spectrum attrihute in the Data class.
- **classification module**: We add a confusion_matrix method according to the 
SRS clarification that returns a confusion matrix for a set of true and predicted
sky object types.  This provides the end user with an additional method to 
evaluate the performance of the object classifier, which offers more information
than an accuracy score alone. 

The data_extraction, preprocessing, and wavelength_alignment modules can be evaluated for integration.