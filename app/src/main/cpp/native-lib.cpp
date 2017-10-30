#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv/modules/calib3d/include/opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

extern "C"
{
double ransac_thresh = 2.5f; // RANSAC inlier threshold
double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
int bb_min_inliers = 30; // Minimal number of inliers to draw bounding box

Mat img_object;
Mat img_scene;
Ptr<ORB> orb;
vector<KeyPoint> keypoints_object;
Mat descriptors_object;
vector<Point2f> object_bb(4);

bool niceHomography(Mat H);
void drawBoundingBox(Mat image, vector<Point2f>);
void readme();

void JNICALL Java_com_example_tim_visiontest_MainActivity_setObjectImage(JNIEnv *env,
                                                                         jobject instance,
                                                                         jlong matAddrInput) {
    img_object = *(Mat *) matAddrInput;
    orb = ORB::create();
    orb->detect(img_object, keypoints_object);
    orb->compute(img_object, keypoints_object, descriptors_object);

    object_bb[0] = Point(0, 0);
    object_bb[1] = Point(img_object.cols, 0);
    object_bb[2] = Point(img_object.cols, img_object.rows);
    object_bb[3] = Point(0, img_object.rows);
    //drawBoundingBox(img_object, object_bb);
}

void JNICALL Java_com_example_tim_visiontest_MainActivity_process(JNIEnv *env, jobject instance,
                                                                  jlong matAddrGray,
                                                                  jlong matAddrRgba) {
    Mat &img_scene = *(Mat *) matAddrGray;
    Mat &colorFrame = *(Mat *) matAddrRgba;
    vector<KeyPoint> keypoints_scene;
    Mat descriptors_scene;
    vector<vector<DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
    vector<Point2f> matched1Point, matched2Point;
    BFMatcher matcher(NORM_HAMMING);

    Mat inlier_mask, homography;
    vector<KeyPoint> inliers1, inliers2;
    vector<DMatch> inlier_matches;

    //cvtColor(frame, img_scene, cv::COLOR_RGB2GRAY);

    orb->detect(img_scene, keypoints_scene);
    orb->compute(img_scene, keypoints_scene, descriptors_scene);

    if (!descriptors_object.empty() && !descriptors_scene.empty()) {
        matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);

        for (unsigned i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
                matched1.push_back(keypoints_object[matches[i][0].queryIdx]);
                matched2.push_back(keypoints_scene[matches[i][0].trainIdx]);

                matched1Point.push_back(keypoints_object[matches[i][0].queryIdx].pt);
                matched2Point.push_back(keypoints_scene[matches[i][0].trainIdx].pt);
            }
        }

        for (unsigned i = 0; i < matched1.size(); i++) {
            circle(colorFrame, matched1[i].pt, 4, Scalar(0, 255, 0), 2);
        }

        if (matched1.size() >= 4) {
            homography = findHomography(matched1Point, matched2Point, RANSAC, ransac_thresh,
                                        inlier_mask);
            if (niceHomography(homography)) {
                for (unsigned i = 0; i < matched1.size(); i++) {
                    if (inlier_mask.at<uchar>(i)) {
                        int new_i = static_cast<int>(inliers1.size());
                        inliers1.push_back(matched1[i]);
                        inliers2.push_back(matched2[i]);
                        inlier_matches.push_back(DMatch(new_i, new_i, 0));
                    }
                }
                vector<Point2f> new_bb;
                perspectiveTransform(object_bb, new_bb, homography);

                if (inliers1.size() >= bb_min_inliers) {
                    drawBoundingBox(colorFrame, new_bb);
                }
            }
        }
        //drawMatches(img_object, inliers1, img_scene, inliers2, inlier_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

}

bool niceHomography(Mat H) {
    const double det =
            H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(1, 0) * H.at<double>(0, 1);
    if (det < 0)
        return false;

    const double N1 = sqrt(
            H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0));
    if (N1 > 4 || N1 < 0.1)
        return false;

    const double N2 = sqrt(
            H.at<double>(0, 1) * H.at<double>(0, 1) + H.at<double>(1, 1) * H.at<double>(1, 1));
    if (N2 > 4 || N2 < 0.1)
        return false;

    const double N3 = sqrt(
            H.at<double>(2, 0) * H.at<double>(2, 0) + H.at<double>(2, 1) * H.at<double>(2, 1));
    if (N3 > 0.002)
        return false;

    return true;
}

void drawBoundingBox(Mat image, vector<Point2f> corners) {
    line(image, corners[0], corners[1], Scalar(0, 255, 0), 4);
    line(image, corners[1], corners[2], Scalar(0, 255, 0), 4);
    line(image, corners[2], corners[3], Scalar(0, 255, 0), 4);
    line(image, corners[3], corners[0], Scalar(0, 255, 0), 4);
}
}