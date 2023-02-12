#!/bin/bash
function gen_input_files(){
echo "Press any key to generate PyTorch input *.py files with seed from 0 to $largest_seed_number in the following path:"
echo $initial_path
read
for i in $(seq 0 $largest_seed_number)
do
  id=$(printf "%05d" $i)
  dirname=${input_file%.py}_$id
  targetfile=${dirname}.py
  targetpbsfile=${dirname}.pbs
  mkdir $dirname
  cp $input_file $dirname/$targetfile
  cp $input_pbs_file $dirname/$targetpbsfile
  sed -i "s/^seed_number = .*  #/seed_number = $i  #/" $dirname/$targetfile
  sed -i "s/^FILENAME=.*\.py  #/FILENAME=$targetfile  #/" $dirname/$targetpbsfile
  echo "'$dirname/$targetfile' is generated!"
done
}

function travel_dir(){
file_counter=0
folder_counter=0
echo
echo "Travelling all subdirectories in the path below to find all *.log files:"
echo $initial_path
for dir in $(ls -R | grep :| tr : " ")
do
  cd $dir
  if ls *.log 1> /dev/null 2>&1;then
    serial=1
    let folder_counter++
	echo
    echo "*** No.$folder_counter folder *** >>> Entered" $dir
	echo "      V V V V"
	for file in *.log
	do
	  echo "[$serial] Locating $file file..."
	  let file_counter++
	  let serial++
	  pick_best      # Process files here
	done
  fi
  cd $initial_path
done
echo
echo "~~~~ Returned to the initial path... ~~~~"
echo $initial_path
echo
  echo "  ######  Total $file_counter *.log PyTorch output files have been extracted  ######"
  echo ">>>>>  Best Model Name: $best_model_name     Best Score: $best_score ($metrics)  <<<<<"
  copy_best_model
echo 
} 

function pick_best(){
if grep -wq "PyTorch for" $file && grep -wq "terminated at" $file;then
  echo "    ==> Results of the test set has been extracted from $file! [$file_counter of $nfiles]"
  if [[ $choice2 == "r" ]];then 	
	  if grep -wq "regression" $file;then
	    CV_loss=$(grep "Average loss on" $file | awk 'END{printf $7}')
	    MAE_test=$(grep "Mean absolute error (MAE) on the test set:" $file | awk 'END{printf $10}')
		MSE_test=$(grep "Mean squared error (MSE) on the test set:" $file | awk 'END{printf $10}')
	    R2_test=$(grep "R-squared (R^2) value on the test set:" $file | awk 'END{printf $9}')
	    echo "        No.$file_counter:    Name: ${file%.log}     Loss (CV)= $CV_loss"
	    echo "        MSE(test)= $MSE_test     MAE(test)= $MAE_test     R^2(test)= $R2_test"
		
	  if [[ $metrics == "CV_loss" && $(echo "$CV_loss < $best_score" | bc) ==  1 ]];then
		best_score=$CV_loss
        best_model_name=${file%.log}
		best_dir=$dir
      elif [[ $metrics == "MAE" && $(echo "$MAE_test < $best_score" | bc) ==  1 ]];then
        best_score=$MAE_test
        best_model_name=${file%.log}
		best_dir=$dir
      elif [[ $metrics == "MSE" && $(echo "$MSE_test < $best_score" | bc) ==  1 ]];then
        best_score=$MSE_test
        best_model_name=${file%.log}
		best_dir=$dir
      elif [[ $metrics == "R^2" && $(echo "$R2_test > $best_score" | bc) ==  1 ]];then
        best_score=$R2_test
        best_model_name=${file%.log}
		best_dir=$dir
      fi
      echo "   *** Current Best Model Name: $best_model_name     Current Best Score: $best_score ($metrics) ***" 
	  else
	    echo "    --> The $file file doesn't contain PyTorch regression outputs! Skipping..."
	    let file_counter--
	  fi
  elif [[ $choice2 == "c" ]];then
    if grep -wq "classification" $file;then
	    CV_loss=$(grep "Average loss on" $file | awk 'END{printf $7}')
	    acc_test=$(grep "Accuracy on the test set:" $file | awk 'END{printf $7}')
      echo "        No.$file_counter:    Name: ${file%.log}     Loss (CV)= $CV_loss"
      echo "        Acc.(test)= $acc_test"
	  
	  if [[ $metrics == "CV_loss" && $(echo "$CV_loss < $best_score" | bc) ==  1 ]];then
		best_score=$CV_loss
		best_model_name=${file%.log}
		best_dir=$dir
      elif [[ $metrics == "accuracy" && $(echo "${acc_test/\%} > ${best_score/\%}" | bc) ==  1 ]];then
        best_score=$acc_test
        best_model_name=${file%.log}
		best_dir=$dir
      fi
      echo "   *** Current Best Model Name: $best_model_name     Current Best Score: $best_score ($metrics) ***" 
	  else
	    echo "    --> The $file file doesn't contain PyTorch classification outputs! Skipping..."
	    let file_counter--
	  fi
  fi
	
else
  echo "    --> The $file file doesn't contain PyTorch outputs! Skipping..."
  let file_counter--
fi  
}

function copy_best_model(){
cd $initial_path
prefix="best_${metrics}_"
best_model_folder_name=${prefix}$best_model_name
cp -r $best_dir $best_model_folder_name
echo "Note: The model with the best $metrics is saved in the '$best_model_folder_name' folder."
}


###################    PyTorch batch generator/picker script begins from the following lines    ################### 
echo
echo "************* PyTorch batch generator/picker of input/output files (*.py/*.log) **************"
echo
initial_path=$(pwd)
echo "Generator or Picker? [g/p]:"
echo "Generator(g): Generate the PyTorch input *.py files with different random seeds based on the same template."
echo "Picker(p): Pick up the PyTorch output *.log files with the best score (e.g. loss(CV), accuracy, MAE, R^2)."
array_choice_1=(g p)
read choice1
while ! echo "${array_choice_1[@]}" | grep -wq "$choice1" 
do
  echo "Please reinput the choice [g/p]..."
  read choice1
done

if [[ $choice1 == "g" ]];then
  echo "Please input a python template file as the PyTorch input file (e.g. pytorch_regression.py):"
  echo "Note: Press enter key to use the default file 'pytorch_regression.py'"
  read input_file
  if [[ -z $input_file ]];then
      input_file='pytorch_regression.py'
  fi
  while [[ ! -f $input_file ]]
  do
    echo "The target template file '$input_file' is not found, please reinput the file:"
    read input_file
	if [[ -z $input_file ]];then
      input_file='pytorch_regression.py'
	fi
  done
  echo "'$input_file' is loaded as the template PyTorch input file!"
  echo
  dos2unix -q ${input_file}
  
  echo "Please input a PBS template file to generate submitting scripts (e.g. pytorch.pbs):"
  echo "Note: Press enter key to use the default file 'pytorch.pbs'"
  read input_pbs_file
  if [[ -z $input_pbs_file ]];then
      input_pbs_file='pytorch.pbs'
  fi
  while [[ ! -f $input_pbs_file ]]
  do
    echo "The target template file '$input_pbs_file' is not found, please reinput the file:"
    read input_pbs_file
	if [[ -z $input_pbs_file ]];then
      input_pbs_file='pytorch.pbs'
	fi
  done
  echo "'$input_pbs_file' is loaded as the PBS template file!"
  echo
  dos2unix -q ${input_pbs_file}
  
  echo "Input the largest seed number (N) to generate input files with {0,1,2,...,N} seed numbers [default:100]:"
  echo "Note: Press enter key to use the default seed number (100)"
  read largest_seed_number
  if [[ -z $largest_seed_number ]];then
      largest_seed_number=100
  fi
  while [[ ! $largest_seed_number =~ ^[1-9][0-9]*$ ]];do
    echo "Please input a positive integer:"
    read largest_seed_number
	if [[ -z $largest_seed_number ]];then
      largest_seed_number=100
    fi
  done
  echo
  gen_input_files

elif [[ $choice1 == "p" ]];then
  nfiles=$(find -name "*.log" | wc -l)
  echo "Please input task type: Regression(r) / Classification(c)? [r/c]:"
  array_choice_2=(r c)
  read choice2
  while ! echo "${array_choice_2[@]}" | grep -wq "$choice2" 
  do
    echo "Please reinput the choice [r/c]..."
    read choice2
  done
  if [[ $choice2 == "r" ]];then
    declare -A map 
    map=(["0"]="CV_loss" ["1"]="MAE" ["2"]="MSE" ["3"]="R^2")
    echo "Which metric is set to pick the best model? [0/1/2/3]"
    echo "[0] Averaged loss on (cross-) validation sets {recommended}"
	echo -e "[1] MAE (test set) \n[2] MSE (test set) \n[3] R^2 (test set)"
    array_choice_3=(0 1 2 3)
    read choice3
    while ! echo "${array_choice_3[@]}" | grep -wq "$choice3" 
    do
      echo "Please reinput the choice [0/1/2/3]..."
      read choice3
    done
    metrics=${map[$choice3]}
  elif [[ $choice2 == "c" ]];then
    declare -A map 
    map=(["0"]="CV_loss" ["1"]="accuracy")
	echo "Which metric is set to pick the best model? [0/1]"
    echo "[0] Averaged loss on (cross-) validation sets {recommended}" 
	echo "[1] accuracy (test set)"
    array_choice_3=(0 1)
    read choice3
    while ! echo "${array_choice_3[@]}" | grep -wq "$choice3" 
    do
      echo "Please reinput the choice [0/1]..."
      read choice3
    done
    metrics=${map[$choice3]}
  fi
  
  echo "Use the '$metrics' metric to pick the best model! Press any key to start..."
  read
  best_model_name=""
  if [[ $metrics == "CV_loss" || $metrics == "MAE" || $metrics == "MSE" ]];then
    best_score=10000000.0
  elif [[ $metrics == "R^2" || $metrics == "accuracy" ]];then
    best_score=0
  fi
  travel_dir
fi







