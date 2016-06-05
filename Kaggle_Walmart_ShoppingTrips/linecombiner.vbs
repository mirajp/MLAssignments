Option Explicit

Dim filename, objFSO, objFile, inputFile, lineread, linetoenter, counter

Set objFSO = CreateObject("Scripting.FileSystemObject")
Set inputFile = objFSO.OpenTextFile("C:\Users\mirag\Downloads\X_testnewlines.txt",1)
Set objFile = objFSO.OpenTextFile("C:\Users\mirag\Downloads\X_newinput.txt",2,true)

Wscript.Echo "Starting..."

lineread = ""
linetoenter = ""
counter = 0

Do While inputFile.AtEndOfStream <> True
    lineread = inputFile.ReadLine
	counter = counter + 1
    linetoenter = linetoenter & " " & lineread
	
	If (counter = 561) Then
		objFile.WriteLine(linetoenter)
		counter = 0
		linetoenter = ""
	End If
Loop

objFile.WriteLine(linetoenter)

inputFile.Close
objFile.Close

Set inputFile = Nothing
Set objFile = Nothing
Set objFSO = Nothing

Wscript.Echo "Done."
Wscript.Quit