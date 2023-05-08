$content = Get-Content "./DataGrabber.ui"

# 替换文本并保存到新文件
$newContent = $content -replace "<pointsize>-1</pointsize>", ""
Set-Content "DataGrabber_fix_n1.ui" $newContent