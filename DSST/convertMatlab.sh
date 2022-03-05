#!/bin/bash
# 功能：将GB2312文件 转换成 UTF-8【解决Windows文件复制到Linux之后乱码问题】
#read -p "Input Path:" SPATH
SPATH="."
#echo $SPATH
POSTFIX="m"
param1="$1"
if [ "$param1" == "win" ];then
   sys1="Linux"
   sys2="Windows"
   format1="UTF-8"
   format2="GB2312"
elif [ "$param1" == "linux" ];then
   sys1="Windows"
   sys2="Linux"
   format1="GB2312"
   format2="UTF-8"
else
   echo "************** 功能 ************"
   echo "  解决matlab脚本文件在Windows和Linux中移动时出现的乱码问题！"
   echo "  将该脚本复制到程序文件夹中，运行该脚本，它会对当前文件夹及子文件夹中的所有*.m文件进行格式转换，解决乱码问题。"
   echo "  转换到 Linux 的命令: $0 linux"
   echo "  转换到 Window的命令: $0 win"
   #exit
fi

echo "********************************"
echo "  格式转换中......"
echo "  从"$sys1"("$format1") 转换到 "$sys2"("$format2")"
echo "********************************"


sys1="Windows"
sys2="Linux"
format1="GB2312"
format2="UTF-8"

FILELIST(){
filelist=`ls $SPATH `
for filename in $filelist; do
	if [ -f $filename ];then
		#echo File:$filename
		#echo "${filename#*.}"
		EXTENSION="${filename#*.}"
		#echo $EXTENSION
		if [ "$EXTENSION" == "$POSTFIX" ];then
		   #echo "${filename%%.*}"
		   echo Processing: $filename
		   iconv -f $format1 -t $format2 $filename -o $filename
		   #iconv -f GB2312 -t UTF-8 $filename -o $filename
		fi

	elif [ -d $filename ];then
		cd $filename
		SPATH=`pwd`
		#echo $SPATH
		FILELIST
		cd ..
	else
		echo "$SPATH/$filename is not a common file."
	fi
done
}
cd $SPATH
FILELIST
echo "======== Convert Done. ========"
