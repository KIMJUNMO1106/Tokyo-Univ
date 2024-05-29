(ros::roseus-add-msgs "jsk_recognition_msgs")

(load "package://spoteus/spot-interface.l")
(spot-init)
(objects (list *spot*))


(send *spot* :reset-pose)
(send *ri* :angle-vector (send *spot* :angle-vector) 8000)
(send *ri* :wait-interpolation)
(send *spot* :arm :angle-vector #f(0.0 -130.0 130.0 90.0 0.0 0.0)) 
(send *ri* :angle-vector (send *spot* :angle-vector) 8000)
(send *ri* :wait-interpolation)
(send *ri* :stop-grasp)
(ros::duration-sleep 2.0)
(send *ri* :start-grasp)
(ros::duration-sleep 1.0)
(send *spot* :arm :angle-vector #f(0.0 -170.0 160.0 90.0 0.0 0.0))
(send *ri* :angle-vector (send *spot* :angle-vector) 8000)
(send *ri* :wait-interpolation)



;; (defvar *topic-name* "/edgetpu_human_pose_estimator/output/poses")  ;; fetch rosbag
(defvar *topic-name* "/edgetpu_panorama_human_pose_estimator/output/poses")


(defun calculate-distance (point1 point2)
  (sqrt (+ (expt (- (elt point1 0) (elt point2 0)) 2)
           (expt (- (elt point1 1) (elt point2 1)) 2)
           (expt (- (elt point1 2) (elt point2 2)) 2))))

(defun right-wrist-position (right-wrist-pose)
  (setq *rw-xyz* (list (send right-wrist-pose :position :x)
                       (send right-wrist-pose :position :y)
                       (send right-wrist-pose :position :z))))

(defun left-wrist-position (left-wrist-pose)
  (setq *lw-xyz* (list (send left-wrist-pose :position :x)
                       (send left-wrist-pose :position :y)
                       (send left-wrist-pose :position :z))))

(defun right-knee-position (right-knee-pose)
  (setq *rk-xyz* (list (send right-knee-pose :position :x)
                       (send right-knee-pose :position :y)
                       (send right-knee-pose :position :z))))

(defun left-knee-position (left-knee-pose)
  (setq *lk-xyz* (list (send left-knee-pose :position :x)
                       (send left-knee-pose :position :y)
                       (send left-knee-pose :position :z))))

(defun get-pose (msg)
  (setq *exist* nil)
  (setq *right-wrist-pose* nil)
  (setq *left-wrist-pose* nil)
  (setq *right-knee-pose* nil)
  (setq *left-knee-pose* nil)
  (setq *people-poses* (send msg :poses))
  (when (not (= (length *people-poses*) 0))
    (setq *people-pose* (elt *people-poses* 0))
    (setq *limb-names* (send *people-pose* :limb_names))
    (setq *limb-poses* (send *people-pose* :poses))

    (dolist (elem (mapcar #'cons *limb-names* *limb-poses*))
      (if (string= (car elem) "right wrist")
	  (setq *right-wrist-pose* (cdr elem))
	))
 
    (dolist (elem (mapcar #'cons *limb-names* *limb-poses*))
      (if (string= (car elem) "left wrist")
	  (setq *left-wrist-pose* (cdr elem))
	))

    (dolist (elem (mapcar #'cons *limb-names* *limb-poses*))
      (if (string= (car elem) "right knee")
	  (setq *right-knee-pose* (cdr elem))
	))

    (dolist (elem (mapcar #'cons *limb-names* *limb-poses*))
      (if (string= (car elem) "left knee")
	  (setq *left-knee-pose* (cdr elem))
	))
    (when (and *right-wrist-pose* *left-wrist-pose* *right-knee-pose* *left-knee-pose*)  
	(setq *exist* t)
      )
    )
)



(setq *count* 0)
(setq *pass_water* nil)

(defun callback (msg)
  (get-pose msg)
  (if *exist*
      (progn
	(right-wrist-position *right-wrist-pose*)
	(left-wrist-position *left-wrist-pose*)
	(right-knee-position *right-knee-pose*)
	(left-knee-position *left-knee-pose*)
	
	(if (and (< (calculate-distance *rk-xyz* *rw-xyz*) 50)
		 (< (calculate-distance *lk-xyz* *lw-xyz*) 50))
	    (progn
	      (print "tired pose")
	      (setq *count* (+ *count* 1))
	      )
	  (progn   
	    (print "B")
	    (setq *count* 0)
	    )
	  )
	(if (> *count* 3)
	    (progn
	      (print "pass the water")
	      (send *spot* :arm  :angle-vector #f(180.0 -60.0 50.0 90.0 0.0 0.0))
	      (send *ri* :angle-vector (send *spot* :angle-vector) 8000)
	      (send *ri* :wait-interpolation)
	      (send *ri* :stop-grasp)
	      (print "passed") 
	      (setq *pass_water* t)
	      )
	  )
	))
  
  )

(ros::roseus "test")
(ros::subscribe *topic-name* jsk_recognition_msgs::PeoplePoseArray #'callback 1)
(ros::rate 100)
(while t
  (if (eq *pass_water* t) (ros::unsubscribe *topic-name*))
  (ros::sleep)
  (ros::spin-once)
  )
