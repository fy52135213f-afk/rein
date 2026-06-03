import pandas as pd 
import numpy as np  
dtype_dict = {"sepallength":  "float64", "sepalwidth":  "float64", "petallength": "float64","petalwidth":"float64","species":"string"}
df = pd.read_csv("iris.csv",sep="\t", dtype=dtype_dict)
array=df.values
array2=array[:,:-1]
array2=array2.astype("float64")
mean=array2.mean(axis=0)
#print(mean)
median=np.median(array2,axis=0)
#print(median)
std=np.std(array2,axis=0)
#print(normalized_array)
x=np.percentile(array2[:,0:1],5,axis=0)
y=np.percentile(array2[:,0:1],95,axis=0)
percentile_array=[x,y]
#print(percentile_array)
array_indices=np.random.choice(array2[:,0:1].size,size=20,replace=False)
array2[:,0:1].flat[array_indices]=np.nan
nan_count=np.isnan(array2[:,0:1]).sum()
nan_indices=np.argwhere(np.isnan(array2[:,0:1]))
#print(nan_indices)
#print(nan_count)
array3=array2[:,0:1][np.where(array2[:,0:1]<5)]
#print(array3)
condition=np.logical_and(array2[:,0]>5,array2[:,2]<1.5)
filtered_array=array[condition]
#print(filtered_array)
condition2=np.logical_and(np.isnan(array2[:,0]),1)
filtered_array2=array2[condition2]
#print(filtered_array2)
array2_clean=np.delete(array2,condition2,axis=0)
#print(result)
correlation_array=np.corrcoef(array2_clean[:,0],array2_clean[:,2])[0,1]
#print(correlation_array)
#print(np.any(np.isnan(array2)))
array4=np.where(np.isnan(array2),0,array2)
#print(array4)
array2_unique,index,count=np.unique(array2[:,0:1],axis=0,return_index=True,return_counts=True)
#print(array2_unique)
#print(count)
condition3=array2[:,3]<np.percentile(array2[:,3],100/3, axis=0)
condition4=array2[:,3]<np.percentile(array2[:,3],200/3, axis=0)
category_array=np.where(condition3,"small",np.where(condition4,"medium","large"))
#print(category_array)
volume_array=(np.pi*array2[:,2]*array2[:,0]**2)/3
volume_array=np.reshape(volume_array,(-1,1))
np.set_printoptions(precision=2,suppress=True)
array2_with_volume=np.concatenate((array2,volume_array),axis=1)
#print(array2_with_volume[:,0:5]) 
np.set_printoptions() 
species=array[:,4]
species_probablity=np.zeros(len(species))
for i in range(len(species)):
    if species[i]=="Iris-setosa":
        species_probablity[i]=0.5
    elif species[i]=="Iris-versicolor":
        species_probablity[i]=0.25
    else:
        species_probablity[i]=0.25
species_probablity=species_probablity/np.sum(species_probablity)
np.random.seed(10086)
random_species_sample=np.random.choice(species,size=100,replace=True,p=species_probablity)
random_species_indices=np.random.choice(species.size,size=100,p=species_probablity)
#random_species_sample=array[random_species_indices]
array_sample=array[random_species_indices]
species,count=np.unique(random_species_sample,return_counts=True)
#print(species,count)
array2_clean_sorted_indices=np.argsort(array2_clean[:,0:1],axis=0)
array2_sorted=array2_clean[array2_clean_sorted_indices]
#print(array2_sorted)
array2_unique,count2=np.unique(array2[:,0:1],axis=0,return_counts=True)
max_count_index=np.argmax(count2)
array2_mode=array2_unique[max_count_index]
#print(array2_mode)
speca_start=np.argwhere(array2[:,3]>1)[0]
print(speca_start)