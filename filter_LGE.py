import pydicom
import matplotlib.pyplot as plt
import pydicom
import os


import json



from pathlib import Path
import shutil

import zipfile

import pdb


# TODO: Create a function that does the following 


SeriesDescriptions = [
    "MDE FLASH PSIR_PSIR",
    "MDE FLASH PSIR_MAG",
    "LATE GAD PSIR SERIES",  # Include but verify
    "LATE GAD MAG SERIES"    # Include but verify
]


# do not look thorugh code, but look at the interface to probe for key functinoalities, i dont need to understand the whole system

"""
def filter_LGE_images(path, destination, SeriesDescription = 'MDE FLASH PSIR_PSIR'):
    
    For each patient dataset combs through each sequence in S1 and determines which attributes does LGE

    Still Outputs a new file directory where the patient folders will be created
    Directory Structure will be maintained -- only retain images that are LGE.

    This will be contained in the SeriesDescription Attribute
    

    # Create a new directory for all the files at destination / *


    # go to the path, build all folders in the directory. 

    # for each file in patient/directory

    dicom = pydicom.dcmread(dicom_dir_path, force = True)
    print(dicom.SeriesDescription) ==  SeriesDescription
    --> add to directory
"""
#SeriesDescriptions = ["MDE FLASH PSIR_PSIR", "MDE FLASH PSIR_MAG", # these ar




def filter_LGE_images(path, destination, SeriesDescriptions= SeriesDescriptions):
    """
    Filters DICOM files based on SeriesDescription and maintains directory structure. This program particularly builds directories only if they contain the series description
    
    Args:
        path: Source directory containing patient folders
        destination: Destination directory for filtered files
        SeriesDescription: The series description to filter for
    
    Example:
    Source:      /base/patient1/s1/dicom1.dcm
                        /patient2/s2/dicom2.dcm
    
    Destination: /dest/patient1_new/s1/dicom1.dcm
                      /patient2_new/s2/dicom2.dcm


    the source directory is the directory that contains the s1 seqeunces 
    """





    # Create destination directory if it doesn't exist
    dest_root = Path(destination)
    dest_root.mkdir(parents=True, exist_ok=True)
    
    # Track which directories we create to remove empty ones later
    created_dirs = set()
    
    # Get the patient directory name (immediate parent of S1, S2, etc.)
    patient_dir = Path(path).name
    new_patient_dir = f"{patient_dir}_new"

    # create a shell direcotry to put the stuff in. 
    dest_path = dest_root / new_patient_dir
    dest_path.mkdir(parents = True, exist_ok = True) 



    
    # Walk through the source directory
    for root, dirs, files in os.walk(path):
    

        for file in files:
            try:
                # Construct full file path
                file_path = Path(root) / file
        
                # Skip non-DICOM files or handle errors
                try:

                    ds = pydicom.dcmread(file_path, force=True)


                    if ds.SeriesDescription not in allSeriesDescriptions: 
                        allSeriesDescriptions.append(str(ds.SeriesDescription))
                    # Check if file has SeriesDescription and matches our filter


                    if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription in SeriesDescriptions:
                        # Calculate relative path to maintain structure
                        
                        rel_path = Path(root).relative_to(path)

                    
                        
                        # Construct destination path with _new appended to patient directory
                        dest_path = dest_root / new_patient_dir / rel_path
                        
                        
                        # Create destination directory
                        dest_path.mkdir(parents=True, exist_ok=True)
                        created_dirs.add(dest_path)
                        
                        # Copy the file
                        shutil.copy2(file_path, dest_path / file)
                        print(f"Copied: {file} to {dest_path}")
                        
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    # Remove empty directories



    print("\nCleaning up empty directories...")
    for dir_path in sorted(created_dirs, reverse=True):  # Process deepest directories first
        try:
            if not any(dir_path.iterdir()):  # If directory is empty
                dir_path.rmdir()
                print(f"Removed empty directory: {dir_path}")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {str(e)}")

    

    return dest_root / new_patient_dir

    
    

    
    



    # after a whoel pass, get rid of empty folders



def explore_dicomdir(dicomdir_path):
    """
    Explore contents of a DICOMDIR file
    """
    # Read the DICOMDIR file
    dicom = pydicom.dcmread(dicomdir_path, force = True)
    
    print(dir(dicom))
    print(dicom.Modality)
    print(dicom.SeriesDescription)
    print(dicom.PatientSex)
    print(dicom.InstitutionName)
    print(dicom.LargestImagePixelValue)
    print(dicom.pixel_array)

    plt.imshow(dicom.pixel_array)

    print(type(dicom.pixel_array))
    plt.show()

    #print(dicom.PixelData)
    """
    # Get the patient records (top level)
    for patient_record in dicomdir.patient_records:
        print("\nPatient:", patient_record.PatientID, patient_record.PatientName)
        
        # Get studies for this patient
        for study in patient_record.children:
            # Print study info
            print("\n  Study:", study.StudyDate, study.StudyDescription)
            
            # Get series in this study
            for series in study.children:
                print(f"\n    Series: {series.SeriesDescription}")
                print(f"    Modality: {series.Modality}")
                
                # Get image instances in this series
                for instance in series.children:
                    # Get the relative path to the image
                    image_path = instance.ReferencedFileID
                    
                    if isinstance(image_path, str):
                        full_path = image_path
                    else:  # it's a sequence of components
                        full_path = os.path.join(*image_path)
                        
                    print(f"      Image: {full_path}")
                    # You can also access other instance-level attributes
                    print(f"      SOP Instance UID: {instance.ReferencedSOPInstanceUIDInFile}")

    """

def unzip_to_folder(zip_path):
    # Get the directory where the zip file is located
    parent_dir = os.path.dirname(zip_path)
    # Get just the filename without extension
    base_name = os.path.splitext(os.path.basename(zip_path))[0]
    # Create full path for extraction
    extract_path = os.path.join(parent_dir, base_name)
    
    # Create the folder if it doesn't exist
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of file names in zip
        file_names = zip_ref.namelist()
        
        # Check if all files are in a single directory that matches the zip name
        common_prefix = os.path.commonprefix(file_names)
        if common_prefix == base_name + '/':
            # Files are already in a directory matching zip name
            # Extract to parent directory instead
            extract_path = parent_dir
            
        zip_ref.extractall(extract_path)
    
    print(f"Unzipped to: {extract_path}")

    return extract_path

def zip_directory(directory_path, output_path=None):
    """
    Zip a directory
    Args:
        directory_path: Path to directory to zip
        output_path: Where to save the zip file. If a directory, will save as directory/foldername.zip
    """
    directory_path = Path(directory_path)
    
    # If output_path is a directory, create zip file with directory name in that location
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir() or str(output_path).endswith('/'):
            output_path = output_path / f"{directory_path.name}.zip"
    else:
        output_path = directory_path.parent / f"{directory_path.name}.zip"
    
    print(f"Creating zip file: {output_path}")
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        # Walk through all files in directory
        for file_path in directory_path.rglob('*'):
            # Calculate path relative to directory being zipped
            relative_path = file_path.relative_to(directory_path)
            
            # Add file to zip
            if file_path.is_file():  # Only zip files, not directories
                print(f"Adding: {relative_path}")
                zip_ref.write(file_path, relative_path)
    
    print(f"Zip file created at: {output_path}")
    return output_path


def get_inner_directory(extract_path):
    # List all items in the directory
    contents = os.listdir(extract_path)
    
    # Filter to get only directories
    directories = [d for d in contents if os.path.isdir(os.path.join(extract_path, d))]
    
    if len(directories) == 1:
        # If there's exactly one directory, return its full path
        inner_dir_path = os.path.join(extract_path, directories[0])
        return inner_dir_path
    elif len(directories) == 2:
        if directories[0] == "__MACOSX": 
            return directories[1]
        elif directories[1] == "__MACOSX":
            return directories[0]
        else: 
            raise Exception("Expected Mac os x file for 2 directories")
    else:
        raise Exception("ERror 3 directories found")
    

if __name__ == '__main__':


    allSeriesDescriptions = []

    # single instance application no need for CLI functionality 
    parent_dir= "./patients/" #shoudl i auto unzip?

    destination = "dest/"

    parent_path = Path(parent_dir)
    
    # Get only immediate directories (does this )
    zip_files = [item for item in parent_path.iterdir() if item.suffix.lower() == '.zip']


    for patient_zip in zip_files: 
        # unzip file to current directory

   

        extract_path = unzip_to_folder(patient_zip)


     
        patient_directory = get_inner_directory(extract_path)


        print(f"patient directory: {patient_directory}")

        # do stuff to the new directory, the zip contaents(direcotry) contains a folder
        new_directory_path = filter_LGE_images(patient_directory, destination)

       
        zip_directory(new_directory_path, destination)

        #  remove unzipped directories both in destination and the original path
        print(extract_path)
        print(new_directory_path)
        shutil.rmtree(extract_path)  # Remove original extracted directory
    
        shutil.rmtree(new_directory_path)  # Remove filtered directory


        
       



    #path = "MR_2016-08-30_5734-0003-1193-7918_raw/"
    #explore_dicomdir(path)
    #filter_LGE_images(path, "./dest")

    print(allSeriesDescriptions)

    with open("SeriesDescriptions.json", "w") as file:
        json.dump(allSeriesDescriptions, file)


    # for each of these patients zip on its own.