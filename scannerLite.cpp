/* 
 * File: scannerLite.cpp
 * Author: daisygao
 * An OpenCV program implementing the recognition feature of the app "CamScanner". 
 * It extracts the main document object from an image and adjust it to A4 size. 
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <vector>
using namespace cv;
using namespace std;

/**
 * Get edges of an image
 * @param gray - grayscale input image
 * @param canny - output edge image
 */
void getCanny(Mat gray, Mat &canny) {
  Mat thres;
  double high_thres = threshold(gray, thres, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU), low_thres = high_thres * 0.5;
  cv::Canny(gray, canny, low_thres, high_thres);
}

struct Line {
  Point _p1;
  Point _p2;
  Point _center;

  Line(Point p1, Point p2) {
    _p1 = p1;
    _p2 = p2;
    _center = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
  }
};

bool cmp_y(const Line &p1, const Line &p2) {
  return p1._center.y < p2._center.y;
}

bool cmp_x(const Line &p1, const Line &p2) {
  return p1._center.x < p2._center.x;
}

/**
 * Compute intersect point of two lines l1 and l2
 * @param l1 
 * @param l2
 * @return Intersect Point
 */
Point2f computeIntersect(Line l1, Line l2) {
  int x1 = l1._p1.x, x2 = l1._p2.x, y1 = l1._p1.y, y2 = l1._p2.y;
  int x3 = l2._p1.x, x4 = l2._p2.x, y3 = l2._p1.y, y4 = l2._p2.y;
  if (float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) {
    Point2f pt;
    pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
    pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
    return pt;
  }
  return Point2f(-1, -1);
}

void scan(String file, bool debug = true) {

  /* get input image */
  Mat img = imread(file);
  // resize input image to img_proc to reduce computation
  Mat img_proc;
  int w = img.size().width, h = img.size().height, min_w = 200;
  double scale = min(10.0, w * 1.0 / min_w);
  int w_proc = w * 1.0 / scale, h_proc = h * 1.0 / scale;
  resize(img, img_proc, Size(w_proc, h_proc));
  Mat img_dis = img_proc.clone();

  /* get four outline edges of the document */
  // get edges of the image
  Mat gray, canny;
  cvtColor(img_proc, gray, CV_BGR2GRAY);
  getCanny(gray, canny);

  // extract lines from the edge image
  vector<Vec4i> lines;
  vector<Line> horizontals, verticals;
  HoughLinesP(canny, lines, 1, CV_PI / 180, w_proc / 3, w_proc / 3, 20);
  for (size_t i = 0; i < lines.size(); i++) {
    Vec4i v = lines[i];
    double delta_x = v[0] - v[2], delta_y = v[1] - v[3];
    Line l(Point(v[0], v[1]), Point(v[2], v[3]));
    // get horizontal lines and vertical lines respectively
    if (fabs(delta_x) > fabs(delta_y)) {
      horizontals.push_back(l);
    } else {
      verticals.push_back(l);
    }
    // for visualization only
    if (debug)
      line(img_proc, Point(v[0], v[1]), Point(v[2], v[3]), Scalar(0, 0, 255), 1, CV_AA);
  }

  // edge cases when not enough lines are detected
  if (horizontals.size() < 2) {
    if (horizontals.size() == 0 || horizontals[0]._center.y > h_proc / 2) {
      horizontals.push_back(Line(Point(0, 0), Point(w_proc - 1, 0)));
    }
    if (horizontals.size() == 0 || horizontals[0]._center.y <= h_proc / 2) {
      horizontals.push_back(Line(Point(0, h_proc - 1), Point(w_proc - 1, h_proc - 1)));
    }
  }
  if (verticals.size() < 2) {
    if (verticals.size() == 0 || verticals[0]._center.x > w_proc / 2) {
      verticals.push_back(Line(Point(0, 0), Point(0, h_proc - 1)));
    }
    if (verticals.size() == 0 || verticals[0]._center.x <= w_proc / 2) {
      verticals.push_back(Line(Point(w_proc - 1, 0), Point(w_proc - 1, h_proc - 1)));
    }
  }
  // sort lines according to their center point
  sort(horizontals.begin(), horizontals.end(), cmp_y);
  sort(verticals.begin(), verticals.end(), cmp_x);
  // for visualization only
  if (debug) {
    line(img_proc, horizontals[0]._p1, horizontals[0]._p2, Scalar(0, 255, 0), 2, CV_AA);
    line(img_proc, horizontals[horizontals.size() - 1]._p1, horizontals[horizontals.size() - 1]._p2, Scalar(0, 255, 0), 2, CV_AA);
    line(img_proc, verticals[0]._p1, verticals[0]._p2, Scalar(255, 0, 0), 2, CV_AA);
    line(img_proc, verticals[verticals.size() - 1]._p1, verticals[verticals.size() - 1]._p2, Scalar(255, 0, 0), 2, CV_AA);
  }

  /* perspective transformation */

  // define the destination image size: A4 - 200 PPI
  int w_a4 = 1654, h_a4 = 2339;
  //int w_a4 = 595, h_a4 = 842;
  Mat dst = Mat::zeros(h_a4, w_a4, CV_8UC3);

  // corners of destination image with the sequence [tl, tr, bl, br]
  vector<Point2f> dst_pts, img_pts;
  dst_pts.push_back(Point(0, 0));
  dst_pts.push_back(Point(w_a4 - 1, 0));
  dst_pts.push_back(Point(0, h_a4 - 1));
  dst_pts.push_back(Point(w_a4 - 1, h_a4 - 1));

  // corners of source image with the sequence [tl, tr, bl, br]
  img_pts.push_back(computeIntersect(horizontals[0], verticals[0]));
  img_pts.push_back(computeIntersect(horizontals[0], verticals[verticals.size() - 1]));
  img_pts.push_back(computeIntersect(horizontals[horizontals.size() - 1], verticals[0]));
  img_pts.push_back(computeIntersect(horizontals[horizontals.size() - 1], verticals[verticals.size() - 1]));

  // convert to original image scale
  for (size_t i = 0; i < img_pts.size(); i++) {
    // for visualization only
    if (debug) {
      circle(img_proc, img_pts[i], 10, Scalar(255, 255, 0), 3);
    }
    img_pts[i].x *= scale;
    img_pts[i].y *= scale;
  }

  // get transformation matrix
  Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);

  // apply perspective transformation
  warpPerspective(img, dst, transmtx, dst.size());

  // save dst img
  imwrite("dst.jpg", dst);

  // for visualization only
  if (debug) {
    namedWindow("dst", CV_WINDOW_KEEPRATIO);
    imshow("src", img_dis);
    imshow("canny", canny);
    imshow("img_proc", img_proc);
    imshow("dst", dst);
    waitKey(0);
  }
}

int main(int argc, char** argv) {
  string img_path[] = {"images/doc1.jpg", "images/doc2.jpg", "images/doc3.jpg"};
  scan(img_path[2]);
  return 0;
}