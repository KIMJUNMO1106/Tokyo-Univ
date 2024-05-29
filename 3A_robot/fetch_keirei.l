(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_recognition_msgs")



(defvar *detected-objects* '())  
(defvar *people-rects* '())      

;;(defun personclose (rect)
;;   (> (* (send rect :width) (send rect :height)) 500)
;;   )


(defun keirei (&optional (tm 10000))
 (let ((torso-rotation (rotate-torso-based-on-rect rect)))
    (send *fetch* :angle-vector (list torso-rotation 150 -70.0 10.0 4.26256e-14 -100.0 170.0 70.0 -40.0 0.0 0.0))
    (send *ri* :angle-vector (send *fetch* :angle-vector) tm)))



(defun update-detected-objects (msg)
  (setq *detected-objects* (send msg :class)))

(defun update-people-rects (msg)
  (setq *people-rects* (send msg :rects)))

(defun person-with-cup-close? ()
  (and (member "cup" *detected-objects*)
       (some #'(lambda (rect)
                 (> (* (send rect :width) (send rect :height)) *threshold-area*))
             *people-rects*)))


 
(defun cb-class (msg)
  (update-detected-objects msg)
  (when (person-with-cup-close?)
    (keirei)
    (return)))


(defun cb-rects (msg)
  (update-people-rects msg))



(ros::subscribe "/edgetpu_object_detector/output/class" jsk_recognition_msgs::ClassificationArray #'cb-class)
(ros::subscribe "/edgetpu_human_pose_estimator/output/rects" jsk_recognition_msgs::RectArray #'cb-rects)


(ros::spin)
