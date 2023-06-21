# DataGrabber

DataGrabber is a tool based on opencv to help you extract data from graph and the demo process is shown below.

![DataGrabber2](https://user-images.githubusercontent.com/15251079/167558979-57febdde-781a-44d8-80dc-b303473ce44e.gif)

## method to use
**Step. 0 Start this program**
This program can be start in the command line by using this line below.
```bash
python main.py
```

**Step. 1 Load graph and input the axis information**

There are two ways to load graph. You can load it from your clipboard or from file. I would recmment you to load graph from clipboard, because it's so convenient. 
Then you need to input the start and end value for both x and y. Then you need to define the graph border. You can click "auto" to find the graph border automatically and of course, you can set the graph border manually if the program fails. 

**Step. 2 Picking up the curve**

When used in **colorful curve extraction**, use the color picker to pick the color from the graph. Selected curve will become white. Then use the eraser to erase unwanted part.
If you are extracting a m**onochrome graph**, drag the morph slider to change the extraction value for the program, you can observe the change and drag it until the curve has been selected correctly. If some unwanted part still extisted, use the eraser!

![image](https://user-images.githubusercontent.com/15251079/167558389-c4c43115-4222-4ceb-bd43-4fa1e33d9373.png)

**Step.3 Save the curve**

Hit "Add" button when current curve has been selected correctly and it requires you to input the name. Then you can repeat step2 to step3 if you want to extract more lines.
Hit "Preview" button to check if everything is fine.
![image](https://user-images.githubusercontent.com/15251079/167558343-5ad81929-22a9-426d-8835-533f737f7331.png)


**Step4. Export the curve**

Click file->export to export your data.

Have fun!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QY7/DataGrabber&type=Date)](https://star-history.com/#QY7/DataGrabber&Date)
