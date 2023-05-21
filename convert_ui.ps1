# 去除 -1 ，否则会有warning
$content = Get-Content "./DataGrabber.ui"

# 替换文本并保存到新文件
$newContent = $content -replace "<pointsize>-1</pointsize>", ""
Set-Content "DataGrabber_fix_n1.ui" $newContent

# pyuic5 -o ./DataGrabber/ui.py ../../Qtdesigner/DataGrabber_v2.ui
pyrcc5 ../../Qtdesigner/Datagrabber.qrc -o ./DataGrabber/Datagrabber_rc.py
pyuic5 -o ./DataGrabber/ui.py ./DataGrabber_fix_n1.ui