## Changes to the API

- **preprocessing module**: We modified the return values of the static functions within 
preprocessing.  Initially the functions normalize, remove_outliers, and correct_redshift
modified the Data class object in place.  However, the SRS clarifications indicate that
each function should be applied to one spectrum, and this alteration makes these
functions more flexible for the user (i.e. they can remove outliers for one spectrum,
but not others).
- **data_extraction module**: We implement a get_spectra method that obtains the spectra 
for the sky objects represented in the Data class. This functionality simplifies
spectra extraction for the end user. 
- **classification module**: We add a confusion_matrix method according to the 
SRS clarification that returns a confusion matrix for a set of true and predicted
sky object types.  This provides the end user with an additional method to 
evaluate the performance of the object classifier, which offers more information
than an accuracy score alone. 