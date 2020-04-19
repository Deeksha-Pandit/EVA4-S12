
<h2>Organisation of JSON file:</h2>

1)	Info - Useful information like year, versio, description, contributor and date created


2)	Images – It contains information about all the images

  •	id – unique id of the image

  •	width – width of the image in pixels

  •	height – height of the image in pixels

  •	file_name – name of the file

  •	license – any license required

  •	date_captured – date of capture


3)	Annotation – This is the manual annotation via VGG

•	id – unique id for annotation

•	image_id – each image has a unique image id

•	segmentation – These are the coordinates of the bounding box (x and y coordinates for the vertices of the rectangle/square around the object)

•	area – area of the bounding box

•	bbox – dimensions of bounding box (top left-x , top right-y, width, height)

•	iscrowd – determines if single or group of images are present. If single image then iscrowd = 0 , if there are group of images then iscrowd =1


4)	License – contains id,name and url


5)	Categories – This describes all the information about the object we have annotated on. Eg: It will contain id,name and other categories.
