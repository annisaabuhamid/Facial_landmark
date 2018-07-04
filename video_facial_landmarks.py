# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
#from sympy.geometry import Point
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
	#help="path to facial landmark predictor")
#ap.add_argument("-r", "--picamera", type=int, default=-1,
	#help="whether or not the Raspberry Pi camera should be used")
#args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		
		shape = face_utils.shape_to_np(shape)

		#leftcheek
		x3 , y3 = tuple(shape[0].ravel())
		x4 ,y4 = tuple(shape[31].ravel())
		mid2= ((x3 + x4)/2),((y4 + y3)/2)
		#eqn
		mid2=(y-y3)/(x-x3) = (y4-y3)/ (x4-x3)
		print(mid2)




		#eqn end

		lc , lc1 = mid2
		lc2, lc3 =tuple(shape[0].ravel())
		lcmid = ((lc + lc2)/2),((lc1 + lc3)/2)
		
		print(lcmid)

		lc4 , lc5 = lcmid
		lc6, lc7 =tuple(shape[31].ravel())
		lcmid1 = ((lc4 + lc6)/2),((lc5 + lc7)/2)

		ax , ax0 = lcmid 
		angle = 360
		cv2.ellipse(frame, mid2, (ax , ax0), angle, 0 , 360, (0, 0, 255), 2)

		#cv2.circle(frame, mid2, 1, (0, 0, 255), 30)


		#rightcheek
		x , y = tuple(shape[16].ravel())
		x1 ,y1 = tuple(shape[35].ravel())
		midpoint= ((x + x1)/2),((y + y1)/2)
		axe1 , axe2 = 27 ,17
		angle = 360
		cv2.ellipse(frame, midpoint, (axe1 , axe2), angle, 0 , 360, (0, 0, 255), 2)

		#cv2.circle(frame, midpoint, 1, (0, 0, 255), 30)

			#bindi
		x5 , y5 = tuple(shape[21].ravel())
		x6 ,y6 = tuple(shape[22].ravel())
		bindi= ((x5 + x6)/2),((y5 + y6)/2)
		cv2.circle(frame, bindi, 1, (0, 0, 255), 3)
		
		#eyelids

		l1 =  tuple(shape[36].ravel())
		l2 =  tuple(shape[37].ravel())
		l3 =  tuple(shape[38].ravel())

		l4 =  tuple(shape[39].ravel())
		l5 =  tuple(shape[40].ravel())
		l6 =  tuple(shape[41].ravel())
		l7 =  tuple(shape[42].ravel())
		l8 =  tuple(shape[43].ravel())
		l9 =  tuple(shape[44].ravel())
		l10 = tuple(shape[45].ravel())
		l11 = tuple(shape[46].ravel())
		l12 = tuple(shape[47].ravel())

		cv2.line(frame, (l1),(l2), (0, 5, 12), 3)
		cv2.line(frame, (l2),(l3), (0, 5, 12), 3)
		cv2.line(frame, (l3),(l4), (0, 5, 12), 3)
		cv2.line(frame, (l4),(l5), (0, 5, 12), 3)
		cv2.line(frame, (l5),(l6), (0, 5, 12), 3)
		cv2.line(frame, (l6),(l1), (0, 5, 12), 3)

		cv2.line(frame, (l7),(l8), (0, 5, 12), 3)
		cv2.line(frame, (l8),(l9), (0, 5, 12), 3)
		cv2.line(frame, (l9),(l10), (0, 5, 12), 3)
		cv2.line(frame, (l10),(l11), (0, 5, 12), 3)
		cv2.line(frame, (l11),(l12), (0, 5, 12), 3)
		cv2.line(frame, (l12),(l7), (0, 5, 12), 3)

		

		
		#lips
		cv2.line(frame, (tuple(shape[48].ravel())), (tuple(shape[49].ravel())), (0, 0, 0),3)
		cv2.line(frame, (tuple(shape[49].ravel())), (tuple(shape[50].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[50].ravel())), (tuple(shape[51].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[59].ravel())), (tuple(shape[48].ravel())), (0, 0, 0), 3)
			
		cv2.line(frame, (tuple(shape[51].ravel())), (tuple(shape[52].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[52].ravel())), (tuple(shape[53].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[53].ravel())), (tuple(shape[54].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[54].ravel())), (tuple(shape[55].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[55].ravel())), (tuple(shape[56].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[56].ravel())), (tuple(shape[57].ravel())), (0, 0, 0), 3)  
			
		cv2.line(frame, (tuple(shape[57].ravel())), (tuple(shape[58].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[58].ravel())), (tuple(shape[59].ravel())), (0, 0, 0), 3)  
			 
		cv2.line(frame, (tuple(shape[60].ravel())), (tuple(shape[61].ravel())), (0, 0, 0), 3)
			
		cv2.line(frame, (tuple(shape[61].ravel())), (tuple(shape[62].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[62].ravel())), (tuple(shape[63].ravel())), (0, 0, 0), 3) 
	  		
	  	cv2.line(frame, (tuple(shape[63].ravel())), (tuple(shape[64].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[64].ravel())), (tuple(shape[65].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[65].ravel())), (tuple(shape[66].ravel())), (0, 0, 0), 3)      
		#print('bottom_lip', (tuple(shape[48].ravel()))
		
		cv2.line(frame, (tuple(shape[66].ravel())), (tuple(shape[67].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[67].ravel())), (tuple(shape[60].ravel())), (0, 0, 0), 3)  
		
		#lefteyebrow
		   
		cv2.line(frame, (tuple(shape[22].ravel())), (tuple(shape[23].ravel())), (0, 0, 0), 5)   
		cv2.line(frame, (tuple(shape[23].ravel())), (tuple(shape[24].ravel())), (0, 0, 0), 5)
		cv2.line(frame, (tuple(shape[24].ravel())), (tuple(shape[25].ravel())), (0, 0, 0), 5)  
		cv2.line(frame, (tuple(shape[25].ravel())), (tuple(shape[26].ravel())), (0, 0, 0), 5)
		
		brow , brow1  =  tuple(shape[22].ravel())
		brow2, brow3 = tuple(shape[42].ravel())
		midbrow = ((brow + brow2)/2),((brow1 + brow3)/2)
		pbrow1 , pbrow2 = midbrow
		pbrow3 , pbrow4 = tuple(shape[22].ravel())
		midbrow2 = ((pbrow1 + pbrow3)/2),((pbrow2 + pbrow4)/2)
		pbrow5 , pbrow6 = midbrow2
		pbrow7 , pbrow8 = tuple(shape[22].ravel())
		fmidbrow3 = ((pbrow5 + pbrow7)/2),((pbrow6 + pbrow8)/2)

		#Second_point
		sbrow , sbrow1 = tuple(shape[23].ravel())
		sbrow2 , sbrow3 = tuple(shape[43].ravel())
		smidbrow = ((sbrow + sbrow2)/2),((sbrow1 + sbrow3)/2)

		sbrow4 , sbrow5 = smidbrow
		sbrow6 , sbrow7 = tuple(shape[23].ravel())
		smidbrow1 = ((sbrow4 + sbrow6)/2),((sbrow5 + sbrow7)/2)

		sbrow8 , sbrow9 = smidbrow1
		sbrow10 , sbrow11 = tuple(shape[23].ravel())
		smidbrow2 = ((sbrow8 + sbrow10)/2),((sbrow9 + sbrow11)/2)

		#third_point

		tbrow , tbrow1 = tuple(shape[24].ravel())
		tbrow2 , tbrow3 = tuple(shape[44].ravel())
		tmidbrow = ((tbrow + tbrow2)/2),((tbrow1 + tbrow3)/2)

		tbrow4 , tbrow5 = tmidbrow
		tbrow6 , tbrow7 = tuple(shape[24].ravel())
		tmidbrow1 = ((tbrow4 + tbrow6)/2),((tbrow5 + tbrow7)/2)

		tbrow8 , tbrow9 = tmidbrow1
		tbrow10 , tbrow11 = tuple(shape[24].ravel())
		tmidbrow2 = ((tbrow8 + tbrow10)/2),((tbrow9 + tbrow11)/2)

		#fourth_point

		fobrow , fobrow1 = tuple(shape[25].ravel())
		fobrow2 , fobrow3 = tuple(shape[44].ravel())
		fomidbrow = ((fobrow + fobrow2)/2),((fobrow1 + fobrow3)/2)

		fobrow4 , fobrow5 = fomidbrow
		fobrow6 , fobrow7 = tuple(shape[25].ravel())
		fomidbrow1 = ((fobrow4 + fobrow6)/2),((fobrow5 + fobrow7)/2)

		fobrow8 , fobrow9 = fomidbrow1
		fobrow10 , fobrow11 = tuple(shape[25].ravel())
		fomidbrow2 = ((fobrow8 + fobrow10)/2),((fobrow9 + fobrow11)/2)

		cv2.line(frame, (tuple(shape[22].ravel())), (midbrow2), (0, 0, 0),5)
		cv2.line(frame, (midbrow2),(smidbrow1) ,(0, 0, 0),5)
		cv2.line(frame, (smidbrow1),(tmidbrow1) ,(0, 0, 0),5)
		cv2.line(frame, (tmidbrow1),(fomidbrow1) ,(0, 0, 0),5)
		cv2.line(frame, (fomidbrow1),(tuple(shape[26].ravel())) ,(0, 0, 0),5)
		
		#right eyebrow
		cv2.line(frame, (tuple(shape[17].ravel())), (tuple(shape[18].ravel())), (0, 0, 0), 5) 
		cv2.line(frame, (tuple(shape[18].ravel())), (tuple(shape[19].ravel())), (0, 0, 0), 5)     
		cv2.line(frame, (tuple(shape[19].ravel())), (tuple(shape[20].ravel())), (0, 0, 0), 5)
		cv2.line(frame, (tuple(shape[20].ravel())), (tuple(shape[21].ravel())), (0, 0, 0), 5)
		#first_right_eyebrow
		rbrow , rbrow1  =  tuple(shape[21].ravel())
		rbrow2, rbrow3 = tuple(shape[39].ravel())
		rmidbrow = ((rbrow + rbrow2)/2),((rbrow1 + rbrow3)/2)
	
		rbrow1 , rbrow2 = rmidbrow
		rbrow3 , rbrow4 = tuple(shape[21].ravel())
		rmidbrow2 = ((rbrow1 + rbrow3)/2),((rbrow2 + rbrow4)/2)

		rbrow5 , rbrow6 = rmidbrow2
		rbrow7 , rbrow8 = tuple(shape[21].ravel())
		rmidbrow3 = ((rbrow5 + rbrow7)/2),((rbrow6 + rbrow8)/2)

		#Second_point
		srbrow , srbrow1 = tuple(shape[20].ravel())
		srbrow2 , srbrow3 = tuple(shape[38].ravel())
		srmidbrow = ((srbrow + srbrow2)/2),((srbrow1 + srbrow3)/2)

		srbrow4 , srbrow5 = srmidbrow
		srbrow6 , srbrow7 = tuple(shape[20].ravel())
		srmidbrow1 = ((srbrow4 + srbrow6)/2),((srbrow5 + srbrow7)/2)

		srbrow8 , srbrow9 = srmidbrow1
		srbrow10 , srbrow11 = tuple(shape[20].ravel())
		srmidbrow2 = ((srbrow8 + srbrow10)/2),((srbrow9 + srbrow11)/2)

		#third_point

		trbrow , trbrow1 = tuple(shape[19].ravel())
		trbrow2 , trbrow3 = tuple(shape[37].ravel())
		trmidbrow = ((trbrow + trbrow2)/2),((trbrow1 + trbrow3)/2)

		trbrow4 , trbrow5 = trmidbrow
		trbrow6 , trbrow7 = tuple(shape[19].ravel())
		trmidbrow1 = ((trbrow4 + trbrow6)/2),((trbrow5 + trbrow7)/2)

		trbrow8 , trbrow9 = trmidbrow1
		trbrow10 , trbrow11 = tuple(shape[19].ravel())
		trmidbrow2 = ((trbrow8 + trbrow10)/2),((trbrow9 + trbrow11)/2)

		#fourth_point

		frobrow , frobrow1 = tuple(shape[18].ravel())
		frobrow2 , frobrow3 = tuple(shape[36].ravel())
		fromidbrow = ((frobrow + frobrow2)/2),((frobrow1 + frobrow3)/2)

		frobrow4 , frobrow5 = fromidbrow
		frobrow6 , frobrow7 = tuple(shape[18].ravel())
		fromidbrow1 = ((frobrow4 + frobrow6)/2),((frobrow5 + frobrow7)/2)

		frobrow8 , frobrow9 = fromidbrow1
		frobrow10 , frobrow11 = tuple(shape[18].ravel())
		fromidbrow2 = ((frobrow8 + frobrow10)/2),((frobrow9 + frobrow11)/2)

		#drawline
		cv2.line(frame, (tuple(shape[21].ravel())), (rmidbrow2), (0, 0, 0), 3)
		cv2.line(frame, (rmidbrow2),(srmidbrow1) ,(0, 0, 0), 3)
		cv2.line(frame, (srmidbrow1),(trmidbrow1) ,(0, 0, 0), 3)
		cv2.line(frame, (trmidbrow1),(fromidbrow1) ,(0, 0, 0), 3)
		cv2.line(frame, (fromidbrow1),(tuple(shape[17].ravel())) ,(0, 0, 0), 3)
		
		
		
		#eyeshadowright
		t , t1  =  tuple(shape[37].ravel())
		t2, t3 = tuple(shape[38].ravel())
		tengah = ((t + t2)/2),((t1 + t3)/2)

		ax1 , ax2 = 40 ,30
		angle = -180
		
		cv2.ellipse(frame, tengah,(ax1 , ax2),angle,0,180,(255,0,0), 2)

		#eyeshadowleft

		h , h1 = tuple(shape[43].ravel())
		h2 , h3 = tuple(shape[44].ravel())
		centre = ((h + h2)/2),((h1 + h3)/2)

		axis1 , axis2 = 37 ,27
		angle = -180
	
		cv2.ellipse(frame, centre,(axis1 , axis2),angle,0,180,(255,0,0), 2)


		#font = cv2.FONT_HERSHEY_SIMPLEX
    	#cv2.putText(frame,'OpenCV',(10,500), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)



    	for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), 0)
		
			
		#extract the leftedgeeye and rightedgeeye with nose (x, y)-coordinates
        #(lStart, lEnd) = FACIAL_LANDMARKS_IDXS[(42,48),]
        #(rStart, rEnd) = FACIAL_LANDMARKS_IDXS[37,32]

        #leftEyeEdgePt, noseLeftEdgePt = shape[lStart:lEnd]
        #rightEyeEdgePt, noseRightEdgePt = shape[rStart:rEnd]

        #Compute the center mass for each eye with nose edge
        #leftEyeEdgeNoseEdge = [(leftEyeEdgePt, noseLeftEdgePt).mean(axis=0).astype("int")]
        #rightEyeEdgeNoseEdge = [(rightEyeEdgePt, noseRightEdgePt).mean(axis=0).astype("int")]

        #compute the angle between the eye and the nose
        #dY = leftEyeEdgePt [1] - noseLeftEdgePt [0]
        #dX = rightEyeEdgePt [1] - noseRightEdgePt [0]
        #angle = np.degrees(np.arctan2(dY,dX)) - 180
        #dist = np.sqrt((dX ** 2) + (dY ** 2))
        #(w,h) = (leftEyeEdgeNoseEdge, rightEyeEdgeNoseEdge)	


		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		#for (w, h) in shape:
			#cv2.circle(frame, (w, h), 1, (0, 0, 255), -1)
	  
	# show the frame
	frame=cv2.flip(frame,1)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 	#time.sleep(60.0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
