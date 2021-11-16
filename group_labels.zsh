#!/bin/zsh

for i in {0..5}
do
	cd dataset
	cat ../labels.csv | grep ",$i" | sed -n -e 's/\(.*\),[0-9]/\1/pg' > label$i.txt
	sed -i -e "s/\r//g" label$i.txt
	rm -f group$i/*
	mkdir -p group$i
	cat label$i.txt | xargs -I {} cp {} group$i
	echo "group$i has $(ls group$i | wc -l) files"
	rm label$i.txt
	rm label$i.txt-e
	cd ..
done
