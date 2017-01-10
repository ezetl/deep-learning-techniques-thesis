#!/usr/bin/env sh

# Script used to split a set of textfiles into validation, test, and train folds
# Pass the files as a command line argument
# Modify the TRAIN and VAL proportions as you whish
# Author: Ezequiel Torti Lopez

TRAIN_PERC=70
VAL_PERC=20
TEST_PERC=$((100 - TRAIN_PERC - VAL_PERC))

echo "Splitting dataset -> Training="$TRAIN_PERC"% Validation="$VAL_PERC"% Testing="$TEST_PERC"%"

# Do not touch this
AUX=aux1.txt
AUX2=aux2.txt
AUX3=aux3.txt
AUX4=aux4.txt

TEST_FILE=test.txt
VAL_FILE=val.txt
TRAIN_FILE=train.txt

touch $AUX

for infile in "$@"
do
    cat $infile >> $AUX
done

echo ""
echo "Processing $(wc -l ${AUX} | awk '{print $1}') lines"

wc -l ${AUX} | awk '{print $1}'>$AUX3
read TOTAL_LINES < $AUX3

TRAIN=$((TOTAL_LINES*TRAIN_PERC/100))
VAL=$((TOTAL_LINES*VAL_PERC/100))
TEST=$((TOTAL_LINES - TRAIN - VAL))

echo ""
echo "Results:"
echo "Training set: "$TRAIN" lines"
echo "Validation set: "$VAL" lines"
echo "Testing set: "$TEST" lines"
echo "Total: " $((TRAIN + VAL + TEST))

# Randomize files
sort -R $AUX > $AUX2

# Get TEST
head -$TEST $AUX2 > $TEST_FILE
tail -$((TOTAL_LINES-TEST)) $AUX2 > $AUX4

# Get VAL
head -$VAL $AUX4 > $VAL_FILE
tail -$((TOTAL_LINES-TEST-VAL)) $AUX2 > $TRAIN_FILE

rm $AUX $AUX2 $AUX3 $AUX4
