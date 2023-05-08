# pyuic5 -o ./DataGrabber/ui.py ../../Qtdesigner/DataGrabber_v2.ui
./replace.ps1
pyrcc5 ../../Qtdesigner/Datagrabber.qrc -o ./DataGrabber/Datagrabber_rc.py
pyuic5 -o ./DataGrabber/ui.py ./DataGrabber_fix_n1.ui