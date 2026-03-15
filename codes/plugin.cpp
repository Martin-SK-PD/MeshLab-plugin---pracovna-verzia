#include "plugin.h"

#include <common/plugins/interfaces/filter_plugin.h>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <queue>
#include <set>

#include <unordered_map>
#include <unordered_set>


#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/stat.h>
#include <vcg/complex/algorithms/update/color.h>
#include <vcg/complex/allocate.h>
#include <vcg/complex/base.h>
#include <vcg/complex/complex.h>
#include <vcg/math/histogram.h>
#include <vcg/math/perlin_noise.h>
#include <vcg/math/random_generator.h>
#include <vcg/space/color4.h>
#include <vcg/space/colormap.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/io_trimesh/import.h>
#include <wrap/callback.h>
#include <wrap/utils.h>

#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QPen>
#include <QTextStream>
#include <QString>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>
#include <QBrush>

#include <chrono>
using namespace std::chrono;



using namespace std;
using namespace tri;
using namespace vcg;

using VertexPtr  = CMeshO::VertexPointer;
using QualityMap = std::unordered_map<VertexPtr, double>;
using AdjMap     = std::unordered_map<VertexPtr, std::vector<VertexPtr>>;


#include <QFile>
#include <QTextStream>
#include <QString>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>

// =========================================================
// ThirdVisLogger
// =========================================================
struct ThirdVisLogger
{
	enum class Mode { Overview = 0, PerVertex = 1 };

	bool               enabled = false;
	Mode               mode    = Mode::Overview;
	QString            filePath; // len pre PerVertex
	std::ostringstream buf;      // bufferujeme (výkon)

	void clear()
	{
		buf.str(std::string());
		buf.clear();
	}

	void line(const std::string& s)
	{
		if (!enabled)
			return;
		buf << s << "\n";
	}

	void text(const std::string& s)
	{
		if (!enabled)
			return;
		buf << s;
	}

	bool flushToFileAppend()
	{
		if (!enabled)
			return true;
		if (filePath.trimmed().isEmpty())
			return false;

		QFile f(filePath);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text))
			return false;

		QTextStream ts(&f);
		ts << QString::fromStdString(buf.str());
		f.close();
		return true;
	}
};

// =========================================================
// RunningStats + formatting helpers
// =========================================================
struct RunningStats
{
	int    n     = 0;
	double sum   = 0.0;
	double sumSq = 0.0;
	double minv  = std::numeric_limits<double>::infinity();
	double maxv  = -std::numeric_limits<double>::infinity();

	void add(double x)
	{
		if (!std::isfinite(x))
			return;
		n++;
		sum += x;
		sumSq += x * x;
		if (x < minv)
			minv = x;
		if (x > maxv)
			maxv = x;
	}

	bool ok() const { return n > 0 && std::isfinite(minv) && std::isfinite(maxv); }

	double mean() const { return (n > 0) ? (sum / (double) n) : 0.0; }

	double stdev() const
	{
		if (n <= 1)
			return 0.0;
		double m   = mean();
		double var = (sumSq / (double) n) - (m * m);
		if (var < 0.0)
			var = 0.0;
		return std::sqrt(var);
	}
};

static inline void
AppendStats(std::ostringstream& oss, const char* name, const RunningStats& s, int prec = 6)
{
	oss.setf(std::ios::fixed);
	oss.precision(prec);

	if (!s.ok()) {
		oss << name << ": (no data)\n";
		return;
	}

	oss << name << ": n=" << s.n << "  min=" << s.minv << "  max=" << s.maxv
		<< "  mean=" << s.mean() << "  stdev=" << s.stdev() << "\n";
}

// Multi-line flush do MeshLab logu po riadkoch.
//prázdne riadky nahradíme " ", aby UI zachovalo odstupy.
static inline void LogMultiline(GLLogStream& log, int level, const std::string& text)
{
	std::istringstream iss(text);
	std::string        line;
	while (std::getline(iss, line)) {
		if (line.empty())
			line = " ";
		log.log(level, line);
	}
}


enum NeighborhoodMode {
	NEIGH_OFF = 0,
	NEIGH_SMOOTH = 1,
	NEIGH_IMPROVE_ONLY = 2
};







// =================== histogram=========================================


// ---------------------------------------------
// Helper for histogram axis scaling
// ---------------------------------------------
static float roundUpToNice(float val)
{
	if (val <= 0.0f || !std::isfinite(val))
		return 1.0f;

	float base = std::pow(10.0f, std::floor(std::log10(val)));
	float n    = val / base;

	if (n <= 1.0f)
		return 1.0f * base;
	else if (n <= 2.0f)
		return 2.0f * base;
	else if (n <= 5.0f)
		return 5.0f * base;
	else
		return 10.0f * base;
}

// ---------------------------------------------
// Build histogram bins from normalized values <0,1>
// ---------------------------------------------
static std::vector<float> BuildHistogramBins(const std::vector<double>& values, int binCount)
{
	std::vector<float> bins(std::max(1, binCount), 0.0f);

	for (double v : values) {
		if (!std::isfinite(v))
			continue;

		if (v < 0.0)
			v = 0.0;
		if (v > 1.0)
			v = 1.0;

		int idx = static_cast<int>(std::floor(v * bins.size()));
		if (idx >= (int) bins.size())
			idx = (int) bins.size() - 1;
		if (idx < 0)
			idx = 0;

		bins[idx] += 1.0f;
	}

	return bins;
}

// ---------------------------------------------
// Save histogram image from already built bins
// ---------------------------------------------
static void saveHistogramImage(const std::vector<float>& data, const std::string& filename)
{
	if (data.empty())
		return;

	const int width  = 800;
	const int height = 400;
	const int margin = 50;

	QImage img(width, height, QImage::Format_ARGB32);
	img.fill(Qt::white);

	QPainter painter(&img);
	painter.setRenderHint(QPainter::Antialiasing);

	painter.setPen(QPen(Qt::black, 2));

	QFont font = painter.font();
	font.setPointSize(8);
	painter.setFont(font);

	float maxVal = *std::max_element(data.begin(), data.end());
	maxVal       = roundUpToNice(maxVal * 1.1f);
	if (maxVal <= 0.0f)
		maxVal = 1.0f;

	int bins = (int) data.size();

	float xScale = float(width - 2 * margin) / bins;
	float yScale = float(height - 2 * margin) / maxVal;

	// axes
	painter.drawLine(margin, height - margin, margin, margin);
	painter.drawLine(margin, height - margin, width - margin, height - margin);

	// Y axis labels
	const int yTicks = 5;
	for (int i = 0; i <= yTicks; ++i) {
		float yValue = i * maxVal / yTicks;
		int   y      = height - margin - int(yValue * yScale);

		painter.drawLine(margin - 5, y, margin + 5, y);

		QString label = QString::number(yValue, 'f', 0);
		painter.drawText(margin - 40, y + 4, label);
	}

	// histogram bars
	painter.setPen(QPen(Qt::blue, 1));
	painter.setBrush(QBrush(QColor(120, 150, 255)));

	for (int i = 0; i < bins; ++i) {
		int x = margin + int(i * xScale);

		int barHeight = int(data[i] * yScale);
		int y         = height - margin - barHeight;

		int barWidth = std::max(1, int(xScale) - 2);

		painter.drawRect(x + 1, y, barWidth, barHeight);
	}

	// X axis labels (0 .. 1 range)
	painter.setPen(Qt::black);

	const int xTicks = 4; // kolko popisov chceme

	for (int i = 0; i <= xTicks; ++i) {
		float value = float(i) / xTicks; // 0..1

		int x = margin + int(value * (width - 2 * margin)); 

		painter.drawLine(x, height - margin - 5, x, height - margin + 5);

		QString label = QString::number(value, 'f', 2);

		painter.drawText(x - 10, height - margin + 20, label);
	}

	img.save(QString::fromStdString(filename));
}

// ---------------------------------------------
// Derive histogram file path from selected log path
// ---------------------------------------------
static QString makeHistogramFilePath(const QString& logFilePath)
{
	QFileInfo fi(logFilePath);
	return fi.absolutePath() + "/" + fi.completeBaseName() + "_hist.png";
}





static double GetMetricOptimalValue(int metricID)
{
	switch (metricID) {
	case 1: return 60.0;            // MaxAngle
	case 2: return 60.0;            // MinAngle
	case 3: return 60.0;            // AvgAngle
	case 4: return 1.0;             // EdgeLengthRatio
	case 5: return std::log1p(6.0); // Valence (regular triangular mesh -> 6 neighbors)
	case 6: return 0.0;             // AngleDeviation360
	case 7: return 0.0;             // MeanCurvature
	default: return 0.0;
	}
}





static double Clamp(double x, double lo, double hi)
{
	if (x < lo)
		return lo;
	if (x > hi)
		return hi;
	return x;
}

static bool IsFinite(double x)
{
	return std::isfinite(x) != 0;
}

// percentil z vektora (p v [0..1])
static double Percentile(std::vector<double>& a, double p)
{
	if (a.empty())
		return 0.0;
	p        = Clamp(p, 0.0, 1.0);
	size_t k = (size_t) std::llround(p * (double) (a.size() - 1));
	std::nth_element(a.begin(), a.begin() + k, a.end());
	return a[k];
}


Plugin::Plugin()
{
	typeList = {FP_FIRST_VIS, FP_SECOND_VIS, FP_Third_VIS};

	for (ActionIDType tt : types())
		actionList.push_back(new QAction(filterName(tt), this));
}

QString Plugin::pluginName() const
{
	return "MetricCombVis - Quality visualization plugin";
}

QString Plugin::filterName(ActionIDType filterId) const
{
	switch (filterId) {
	case FP_FIRST_VIS: return QString("MetricCombVis: First visualization");
	case FP_SECOND_VIS: return QString("MetricCombVis: Second visualization");
	case FP_Third_VIS: return QString("MetricCombVis: Third visualization");
	default: assert(0); return QString();
	}
}

QString Plugin::pythonFilterName(ActionIDType f) const
{
	switch (f) {
	case FP_FIRST_VIS: return QString("");
	case FP_SECOND_VIS: return QString("");
	case FP_Third_VIS: return QString("");
	default: assert(0); return QString();
	}
}

QString Plugin::filterInfo(ActionIDType filterId) const
{
	switch (filterId) {
	case FP_FIRST_VIS:
		return tr(
			"This function allows users to apply a combination of one to five quality metrics to a "
			"mesh.<br>"
			"The user also has the option to set how colors will be mapped to individual faces.");
	case FP_SECOND_VIS:
		return tr(
			"This function allows users to apply colors to vertices according to the average of "
			"the angles"
			" that are around the vertex, or according to the maximum deviation from the optimal, "
			"60° angle.<br>"
			"Before applying or previewing this function, enable option of showing vertices, and "
			"optionally also "
			"option of showing edges of the mesh in the toolbar.<br>"
			"After applying or previewing the function, "
			"set the color of the faces in the side panel to 'Mesh' or 'User-Def'.<br>"
			"In the side panel, you can also set the size of the points that represent the "
			"vertices.");
	case FP_Third_VIS:
		return tr("");
	default: assert(0);
	}
	return "";
}

RichParameterList Plugin::initParameterList(const QAction* a, const MeshDocument& md)
{
	RichParameterList parlst;
	switch (ID(a)) {
	case FP_FIRST_VIS: {


		QStringList metrics;
		metrics.push_back("inradius/circumradius");
		metrics.push_back("Area");
		metrics.push_back("MinAndMaxAngle");
		metrics.push_back("MaxMinSideRatio");
		metrics.push_back("MinMaxHeights");

		parlst.addParam(RichEnum(
			"metric1",
			0,
			metrics,
			tr("Metric1:"),
			tr("Choose a metric to compute triangle quality.")));

		metrics.push_back("None");
		parlst.addParam(RichEnum(
			"metric2",
			5,
			metrics,
			tr("Metric2:"),
			tr("Choose a metric to compute triangle quality.")));
		parlst.addParam(RichEnum(
			"metric3",
			5,
			metrics,
			tr("Metric3:"),
			tr("Choose a metric to compute triangle quality.")));
		parlst.addParam(RichEnum(
			"metric4",
			5,
			metrics,
			tr("Metric4:"),
			tr("Choose a metric to compute triangle quality.")));
		parlst.addParam(RichEnum(
			"metric5",
			5,
			metrics,
			tr("Metric5:"),
			tr("Choose a metric to compute triangle quality.")));

		QStringList mix;
		mix.push_back("Use metric specific values");
		mix.push_back("Use range values and average value as optimal");
		mix.push_back("Use range values and middle middle value of range as optimal");
		parlst.addParam(RichEnum(
			"colorMixFaktor",
			0,
			mix,
			tr("Color Distribution:"),
			tr("Choose a color distribution method.<br>"
			   "The first option uses the optimal values ​​of the given visualization.<br>"
			   "The second option uses the range of calculated values ​​and uses the average "
			   "of all values ​​as the optimal value.<br>"
			   "The third option uses the range of calculated values ​​and uses the average of "
			   "the min and max values ​​as the optimal value.")));
		break;
	}
	case FP_SECOND_VIS: {
		QStringList mix1;
		mix1.push_back("Use average of vertex angles colors");
		mix1.push_back("Use max diference from optimal angle");

		parlst.addParam(RichEnum(
			"colorMixFaktor1",
			0,
			mix1,
			tr("Color Distribution:"),
			tr("Choose a color distribution method.")));
		break;
	}
	case FP_Third_VIS: {
		QStringList vertexMetrics;
		vertexMetrics.push_back("None");
		vertexMetrics.push_back("MaxAngle");
		vertexMetrics.push_back("MinAngle");
		vertexMetrics.push_back("AvgAngle");
		vertexMetrics.push_back("EdgeLengthRatio");
		vertexMetrics.push_back("Valence");
		vertexMetrics.push_back("AngleDeviation360");
		vertexMetrics.push_back("MeanCurvature");

		parlst.addParam(RichEnum(
			"vertexMetricA",
			3,
			vertexMetrics,
			"Primary Metric",
			"Select first vertex quality metric."));
		parlst.addParam(RichEnum(
			"vertexMetricB",
			0,
			vertexMetrics,
			"Secondary Metric",
			"Select second metric (or None)."));
		parlst.addParam(RichDynamicFloat(
			"metricMixRatio",
			0.5,
			0,
			1,
			"Metric Mix (0 = B only, 1 = A only)",
			"Blend ratio between metrics A and B." ));

		QStringList normalizationModes;
		normalizationModes.push_back("Range-based");
		normalizationModes.push_back("Optimality-based");
		parlst.addParam(RichEnum(
			"normalizationMode",
			0,
			normalizationModes,
			"Normalization mode",
			"Range-based uses the observed value range. "
			"Optimality-based maps the metric optimum to the middle of the interval (0.5)."));


		parlst.addParam(RichInt(
			"neighborhoodRadius",
			0, // 0 = vypnuté
			"Neighborhood radius",
			"How many edge-steps from the vertex are included (0 = only the vertex)."));

		QStringList neighModes;
		neighModes.push_back("Off");
		neighModes.push_back("Smooth");
		neighModes.push_back("ImproveOnly");
		parlst.addParam(RichEnum(
			"neighborhoodMode",
			0,
			neighModes,
			"Neighborhood mode",
			"Neighborhood-based post-processing of vertex scores."));

		parlst.addParam(RichDynamicFloat(
			"neighborhoodWeight",
			0.5,
			0.0,
			1.0,
			"Neighborhood weight",
			"0 = keep original score, 1 = use only neighborhood score."));

		// --- Logging for Third visualization ---
		parlst.addParam(RichBool(
			"enableLogging",
			false,
			"Enable logging",
			"If enabled, the filter will generate debug output (may slow down processing)."));

		QStringList logVerb;
		logVerb.push_back("Overview (MeshLab Log)");
		logVerb.push_back("Per-vertex (File)");
		parlst.addParam(RichEnum(
			"logVerbosity",
			0,
			logVerb,
			"Log verbosity",
			"Overview writes a short summary into MeshLab Log. "
			"Per-vertex writes one line per vertex into a file (slow)."));

		// Save-file dialog (only used for Per-vertex)
		parlst.addParam(RichFileSave(
			"logOutputFile",
			"",
			"*.txt",
			"Save per-vertex log as",
			"Used only if Log verbosity = Per-vertex (File)."));
		parlst.addParam(RichBool(
			"saveHistogramImage",
			false,
			"Save histogram image",
			"If enabled and Per-vertex file logging is used, a histogram image "
			"will be saved next to the selected log file."));
		break;
	}
	default: break;
	}
	return parlst;
}

FilterPlugin::FilterClass Plugin::getClass(const QAction* a) const
{
	switch (ID(a)) {
	case FP_FIRST_VIS: return FilterPlugin::Quality;
	case FP_SECOND_VIS: return FilterPlugin::Quality;
	case FP_Third_VIS : return FilterPlugin::Quality;
	default: assert(0); return FilterPlugin::Generic;
	}
}

QString Plugin::filterScriptFunctionName(ActionIDType filterID)
{
	switch (filterID) {
	case FP_FIRST_VIS: return QString("First Vis");
	case FP_SECOND_VIS: return QString("Second Vis");
	case FP_Third_VIS: return QString("Third Vis");
	default: assert(0);
	}
	return NULL;
}

bool Plugin::AreAllFacesTriangles(CMeshO& mesh)
{
	int faceIndex = 0; // To keep track of the face index for detailed debugging.

	for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi, ++faceIndex) {
		if (!(*fi).IsD()) { // Check if the face is not deleted
			if ((*fi).VN() != 3) {
				std::cout << "Non-triangular face at index " << faceIndex
						  << " with vertex count: " << (*fi).VN() << std::endl;
				return false; // Found a face that is not a triangle
			}

			// Check if all vertices of the triangle are distinct
			if ((*fi).V(0) == (*fi).V(1) || (*fi).V(1) == (*fi).V(2) || (*fi).V(2) == (*fi).V(0)) {
				std::cout << "Degenerate triangle at index " << faceIndex
						  << " (coincident vertices)." << std::endl;
				return false;
			}
		}
	}

	return true; // All faces are triangles
}

// Function to calculate the area of a triangle
double Plugin::CalculateTriangleArea(CMeshO::FaceIterator fi)
{
	Point3f& v1            = (*fi).V(0)->P();
	Point3f& v2            = (*fi).V(1)->P();
	Point3f& v3            = (*fi).V(2)->P();
	Point3f  edge1         = v2 - v1;
	Point3f  edge2         = v3 - v1;
	Point3f  cross_product = edge1 ^ edge2; // Cross product of edge1 and edge2
	double   area =
		0.5 *
		cross_product.Norm(); // Norm of the cross product gives twice the area of the triangle
	return area;
}

// Manual calculation of the Euclidean distance between two 3D points
double Plugin::CalculateDistance(const vcg::Point3f& p1, const vcg::Point3f& p2)
{
	double dx = p2.X() - p1.X();
	double dy = p2.Y() - p1.Y();
	double dz = p2.Z() - p1.Z();
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Function to calculate and return ratio of minimum and maximum height in a triangle
double Plugin::CalculateMinMaxHeightsRatio(CMeshO::FaceIterator fi)
{
	if (!(*fi).IsD()) {
		vcg::Point3f v1 = (*fi).V(0)->P();
		vcg::Point3f v2 = (*fi).V(1)->P();
		vcg::Point3f v3 = (*fi).V(2)->P();

		double area = CalculateTriangleArea(fi);

		// Calculate the length of each side of the triangle manually
		double lengthA = CalculateDistance(v2, v3); // Length of side BC
		double lengthB = CalculateDistance(v1, v3); // Length of side AC
		double lengthC = CalculateDistance(v1, v2); // Length of side AB

		// Calculate height from each vertex to the opposite side
		double heightA = (2 * area) / lengthA; // Height from vertex A to side BC
		double heightB = (2 * area) / lengthB; // Height from vertex B to side AC
		double heightC = (2 * area) / lengthC; // Height from vertex C to side AB

		// Find min and max heights
		double minHeight = std::min({heightA, heightB, heightC});
		double maxHeight = std::max({heightA, heightB, heightC});

		double ratio = minHeight / maxHeight;

		if (isnan(ratio)) {
			return 0.0;
		}

		return ratio;
	}

	return -1.0; // Return invalid values if the face is deleted or another error occurs
}

// Function to calculate and return the ratio of the maximum length side to the minimum length side
// in a triangle
double Plugin::CalculateMaxMinSideRatio(CMeshO::FaceIterator fi)
{
	if (!(*fi).IsD()) {
		vcg::Point3f v1 = (*fi).V(0)->P();
		vcg::Point3f v2 = (*fi).V(1)->P();
		vcg::Point3f v3 = (*fi).V(2)->P();

		// Calculate the length of each side of the triangle
		double lengthA = CalculateDistance(v1, v2);
		double lengthB = CalculateDistance(v2, v3);
		double lengthC = CalculateDistance(v3, v1);

		// Find the maximum and minimum side lengths
		double maxLength = std::max({lengthA, lengthB, lengthC});
		double minLength = std::min({lengthA, lengthB, lengthC});

		// Compute and return the ratio of the maximum length to the minimum length
		if (minLength > 0) { // Ensure no division by zero
			return maxLength / minLength;
		}
		else {
			0; // Handle cases where minimum length is zero
		}
	}

	return -1.0; // Return an invalid value if the face is deleted or another error occurs
}

// Function to compute ratio of the radius of circumscribed and inscribed circles
double Plugin::CalculateCircleRadii(CMeshO::FaceIterator fi)
{
	if (!(*fi).IsD()) {
		vcg::Point3f v1 = (*fi).V(0)->P();
		vcg::Point3f v2 = (*fi).V(1)->P();
		vcg::Point3f v3 = (*fi).V(2)->P();

		double a    = CalculateDistance(v1, v2);
		double b    = CalculateDistance(v2, v3);
		double c    = CalculateDistance(v3, v1);
		double area = CalculateTriangleArea(fi);
		double s    = (a + b + c) / 2.0; // Semi-perimeter

		if (area == 0 || s == 0) {
			return 0;
		}

		double circumRadius = (a * b * c) / (4.0 * area);
		double inRadius     = area / s;

		if (circumRadius == 0) {
			return 0; // Return an invalid value if inRadius is zero to avoid division by zero
		}

		double ratio = inRadius / circumRadius;

		return ratio;
	}
	return 0;
}


bool Plugin::isNegativeNaN(double x)
{
	if (std::isnan(x)) {
		uint64_t bits;
		std::memcpy(&bits, &x, sizeof(bits));
		return bits >> 63; // Check the sign bit
	}
	return false;
}




bool Plugin::isValidNumber(double x)
{
	if (isnan(x))
		return false;

	if (isNegativeNaN(x))
		return false;

	// zvyšok NaN testov (x != x je univerzálny NaN check)
	if (x != x)
		return false;

	// veľmi veľké absolútne hodnoty považujeme za overflow/INF
	if (fabs(x) > 1e300)
		return false;

	return true;
}

bool Plugin::isValidPoint(const vcg::Point3f& p)
{
	return isValidNumber(p.X()) && isValidNumber(p.Y()) && isValidNumber(p.Z());
}



void Plugin::BuildVertexFaceAdjacency(CMeshO& mesh)
{
	m_vertFaceAdj.clear();

	for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
		if (fi->IsD())
			continue;

		CMeshO::FacePointer fp = &*fi;

		CMeshO::VertexPointer v0 = fi->V(0);
		CMeshO::VertexPointer v1 = fi->V(1);
		CMeshO::VertexPointer v2 = fi->V(2);

		// základné sanity
		if (!v0 || !v1 || !v2)
			continue;
		if (v0->IsD() || v1->IsD() || v2->IsD())
			continue;

		// degenerované tvary preskočíme
		if (v0 == v1 || v1 == v2 || v2 == v0)
			continue;

		m_vertFaceAdj[v0].push_back(fp);
		m_vertFaceAdj[v1].push_back(fp);
		m_vertFaceAdj[v2].push_back(fp);
	}
}





// Function to calculate angles in a triangle using the Law of Cosines, return angle with biggest
// diference to 60deg
double Plugin::CalculateMinMaxAngles(CMeshO::FaceIterator fi)
{
	if (!(*fi).IsD()) {
		vcg::Point3f v1 = (*fi).V(0)->P();
		vcg::Point3f v2 = (*fi).V(1)->P();
		vcg::Point3f v3 = (*fi).V(2)->P();

		// Lengths of the sides of the triangle
		double a = CalculateDistance(v2, v3);
		double b = CalculateDistance(v1, v3);
		double c = CalculateDistance(v1, v2);

		// Angles in radians using the Law of Cosines
		double angleA = std::acos((b * b + c * c - a * a) / (2 * b * c));
		double angleB = std::acos((a * a + c * c - b * b) / (2 * a * c));
		double angleC = std::acos((a * a + b * b - c * c) / (2 * a * b));

		// Convert radians to degrees
		double angleADeg = angleA * (180.0 / M_PI);
		double angleBDeg = angleB * (180.0 / M_PI);
		double angleCDeg = angleC * (180.0 / M_PI);

		// Find minimum and maximum angles
		double minAngle = std::min({angleADeg, angleBDeg, angleCDeg});
		double maxAngle = std::max({angleADeg, angleBDeg, angleCDeg});

		if (minAngle <= 0 || isNegativeNaN(minAngle)) {
			return 0;
		}
		if (maxAngle >= 180 || isnan(maxAngle)) {
			return 0;
		}

		if ((60 - minAngle) <= (maxAngle - 60)) {
			return maxAngle;
		}
		else {
			return minAngle;
		}
	}

	return -1; // Return invalid values if the face is deleted or another error occurs
}

// Function to interpolate between two colors based on a factor
Color4b Plugin::InterpolateColor(
	const vcg::Color4b& colorStart,
	const vcg::Color4b& colorEnd,
	double              factor)
{
	return vcg::Color4b(
		colorStart[0] + (colorEnd[0] - colorStart[0]) * factor,
		colorStart[1] + (colorEnd[1] - colorStart[1]) * factor,
		colorStart[2] + (colorEnd[2] - colorStart[2]) * factor,
		255 // Set alpha to maximum for full opacity
	);
}

// Function to get a color based on a value, relative to a minimum, optimal, and maximum
Color4b Plugin::GetColorForValue(double value, double min, double optimal, double max)
{
	// Bezpečnostné kontroly
	// (odporúčam mať hore v súbore aj #include <cmath>)
	if (!std::isfinite(value))
		value = optimal;

	if (!std::isfinite(min) || !std::isfinite(optimal) || !std::isfinite(max)) {
		// niečo je úplne zle -> neutrálny výsledok
		return vcg::Color4b(0, 255, 0, 255);
	}

	// Ak je rozsah zdegenerovaný, vráť neutrálnu zelenú
	if (max <= min) {
		return vcg::Color4b(0, 255, 0, 255);
	}

	// Uisti sa, že optimal je v rozsahu <min, max>
	if (optimal < min)
		optimal = min;
	if (optimal > max)
		optimal = max;

	// klasické orezanie mimo rozsah
	if (value < min)
		value = min;
	if (value > max)
		value = max;

	// Ak je hodnota presne optimálna
	if (std::fabs(value - optimal) < 1e-12)
		return vcg::Color4b(0, 255, 0, 255); // Green for optimal

	// Interpolácia
	if (value < optimal) {
		// Interpolácia medzi červenou a zelenou
		double denom  = (optimal - min);
		double factor = (denom > 1e-9) ? (value - min) / denom : 0.0;
		if (!std::isfinite(factor))
			factor = 0.0;
		factor = std::max(0.0, std::min(1.0, factor));

		return InterpolateColor(vcg::Color4b(255, 0, 0, 255), vcg::Color4b(0, 255, 0, 255), factor);
	}
	else {
		// Interpolácia medzi zelenou a modrou
		double denom  = (max - optimal);
		double factor = (denom > 1e-9) ? (value - optimal) / denom : 0.0;
		if (!std::isfinite(factor))
			factor = 0.0;
		factor = std::max(0.0, std::min(1.0, factor));

		return InterpolateColor(vcg::Color4b(0, 255, 0, 255), vcg::Color4b(0, 0, 255, 255), factor);
	}
}


Color4b Plugin::AverageColors(const vcg::Color4b& color1, const vcg::Color4b& color2)
{
	unsigned char red   = (color1[0] + color2[0]) / 2;
	unsigned char green = (color1[1] + color2[1]) / 2;
	unsigned char blue  = (color1[2] + color2[2]) / 2;
	unsigned char alpha =
		(color1[3] + color2[3]) / 2; // Assuming we also want to average the alpha channel

	return vcg::Color4b(red, green, blue, alpha);
}


void Plugin::BlendFaceColor(CMeshO::FaceIterator fi, const vcg::Color4b& blendColor)
{
	if (!(*fi).IsD()) {                      // Check if the face is not deleted
		vcg::Color4b currentColor = fi->C(); // Get the current color of the face
		vcg::Color4b newColor     = AverageColors(currentColor, blendColor);
		fi->C()                   = newColor; // Set the new averaged color to the face
	}
}








double Plugin::ComputeVertexMetric(int metricID, CMeshO::VertexPointer v, CMeshO& mesh)
{
	if (!v || v->IsD())
		return 0.0;

	const vcg::Point3f& pv = v->P();
	if (!isValidPoint(pv))
		return 0.0;

	std::vector<double> angles;
	std::vector<double> edgeLengths;

	// >>> namiesto prechádzania všetkých facov použijeme m_vertFaceAdj <<<
	auto itFaces = m_vertFaceAdj.find(v);
	if (itFaces == m_vertFaceAdj.end()) {
		// vertex nemá žiadne incidentné facy
		return 0.0;
	}

	const std::vector<CMeshO::FacePointer>& faces = itFaces->second;

	for (CMeshO::FacePointer fp : faces) {
		if (!fp || fp->IsD())
			continue;

		CMeshO::VertexPointer fv[3];
		bool                  badFace = false;

		for (int i = 0; i < 3; ++i) {
			fv[i] = fp->V(i);
			if (!fv[i] || fv[i]->IsD() || !isValidPoint(fv[i]->P())) {
				badFace = true;
				break;
			}
		}
		if (badFace)
			continue;

		// degenerované trojuholníky
		if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0])
			continue;

		// nájsť index nášho vertexu v tejto face
		int idx = -1;
		for (int i = 0; i < 3; ++i)
			if (fv[i] == v) {
				idx = i;
				break;
			}

		if (idx == -1)
			continue;

		CMeshO::VertexPointer v1 = fv[(idx + 1) % 3];
		CMeshO::VertexPointer v2 = fv[(idx + 2) % 3];

		const vcg::Point3f& p1 = v1->P();
		const vcg::Point3f& p2 = v2->P();

		if (!isValidPoint(p1) || !isValidPoint(p2))
			continue;

		// dĺžky hrán
		double a = CalculateDistance(p1, p2); // hrana medzi susedmi
		double b = CalculateDistance(pv, p2); // hrana od v
		double c = CalculateDistance(pv, p1); // hrana od v

		if (!isValidNumber(a) || !isValidNumber(b) || !isValidNumber(c))
			continue;

		if (a <= 0.0 || b <= 0.0 || c <= 0.0)
			continue;

		// trojuholníková nerovnosť
		if (a + b <= c || a + c <= b || b + c <= a)
			continue;

		// zákon kosínov
		double denom = 2.0 * b * c;
		if (!isValidNumber(denom) || fabs(denom) < 1e-20)
			continue;

		double num = (b * b + c * c - a * a);
		if (!isValidNumber(num))
			continue;

		double cosAngle = num / denom;
		if (!isValidNumber(cosAngle))
			continue;

		if (cosAngle < -1.0)
			cosAngle = -1.0;
		if (cosAngle > 1.0)
			cosAngle = 1.0;

		double angle = acos(cosAngle) * 180.0 / M_PI;
		if (!isValidNumber(angle))
			continue;

		angles.push_back(angle);

		// dĺžky hrán použiteľné pre ratio
		double l1 = CalculateDistance(pv, p1);
		double l2 = CalculateDistance(pv, p2);

		if (isValidNumber(l1) && l1 > 0.0)
			edgeLengths.push_back(l1);
		if (isValidNumber(l2) && l2 > 0.0)
			edgeLengths.push_back(l2);
	}

	double result = 0.0;

	switch (metricID) {
	case 1: // MaxAngle
		if (!angles.empty())
			result = *std::max_element(angles.begin(), angles.end());
		break;

	case 2: // MinAngle
		if (!angles.empty())
			result = *std::min_element(angles.begin(), angles.end());
		break;

	case 3: // AvgAngle
		if (!angles.empty()) {
			double sum = std::accumulate(angles.begin(), angles.end(), 0.0);
			if (isValidNumber(sum))
				result = sum / angles.size();
		}
		break;

	case 4: // EdgeLengthRatio
		if (!edgeLengths.empty()) {
			double maxL = *std::max_element(edgeLengths.begin(), edgeLengths.end());
			double minL = *std::min_element(edgeLengths.begin(), edgeLengths.end());
			if (isValidNumber(maxL) && isValidNumber(minL) && minL > 0.0)
				result = maxL / minL;
		}
		break;

	case 5: { // Valence (počet susedných facov)
		int count = (int) faces.size();
		result    = log1p((double) count);
		if (!isValidNumber(result))
			result = 0.0;
		break;
	}

	case 6: { // AngleDeviation360
		if (!angles.empty()) {
			double sum = std::accumulate(angles.begin(), angles.end(), 0.0);
			if (isValidNumber(sum))
				result = fabs(sum - 360.0);
		}
		break;
	}
	case 7: { // MeanCurvature
		result = ComputeMeanCurvature(v, mesh);
		break;
	}

	default: result = 0.0;
	}


	if (!isValidNumber(result))
		result = 0.0;

	return result;
}






void Plugin::FP_FIRST_VIS_Apply(
	CMeshO& mesh,
	int     metric,
	int     averageOfSpecific_or_UseRanges,
	bool    mixColors)
{
	if (averageOfSpecific_or_UseRanges == 0) {
		switch (metric) {
		// -------------------------
		// 0) CIRCUM/INRADIUS RATIO
		// -------------------------
		case 0: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi)) {
					double a = CalculateCircleRadii(fi);

					if (mixColors) {
						Color4b new_color = GetColorForValue(a, 0, 0.6, 1);
						BlendFaceColor(fi, new_color);
					}
					else {
						fi->C() = GetColorForValue(a, 0, 0.6, 1);
					}
				}
			}
		} break;

		// -------------------------
		// 1) TRIANGLE AREA
		// -------------------------
		case 1: {
			// Compute + store
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				double area = 0;
				if (!fi->IsD() && (area = CalculateTriangleArea(fi)) != 0) {
					CMeshO::FacePointer fp = &*fi;

					data_struct.sum += area;
					data_struct.counter++;
					data_struct.max = std::max(data_struct.max, area);
					data_struct.min = std::min(data_struct.min, area);

					data_struct.map[fp] = area;
				}
			}

			// Colorize
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (fi->IsD())
					continue;

				CMeshO::FacePointer fp = &*fi;
				if (data_struct.map.find(fp) == data_struct.map.end())
					continue;

				double val = data_struct.map.at(fp);

				if (mixColors) {
					Color4b newColor = GetColorForValue(
						val,
						data_struct.min,
						data_struct.sum / data_struct.counter,
						data_struct.max);
					BlendFaceColor(fi, newColor);
				}
				else {
					fi->C() = GetColorForValue(
						val,
						data_struct.min,
						data_struct.sum / data_struct.counter,
						data_struct.max);
				}
			}
		} break;

		// -------------------------
		// 2) MIN/MAX ANGLE
		// -------------------------
		case 2: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					double angle = CalculateMinMaxAngles(fi);

					if (mixColors)
						BlendFaceColor(fi, GetColorForValue(angle, 0, 60, 180));
					else
						fi->C() = GetColorForValue(angle, 0, 60, 180);
				}
			}
		} break;

		// -------------------------
		// 3) MAX/MIN SIDE RATIO
		// -------------------------
		case 3: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					double r = CalculateMaxMinSideRatio(fi);

					if (mixColors)
						BlendFaceColor(fi, GetColorForValue(r, 0, 1, 3));
					else
						fi->C() = GetColorForValue(r, 0, 1, 3);
				}
			}
		} break;

		// -------------------------
		// 4) MIN/MAX HEIGHT RATIO
		// -------------------------
		case 4: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					double r = CalculateMinMaxHeightsRatio(fi);

					if (mixColors)
						BlendFaceColor(fi, GetColorForValue(r, 0.25, 1, 2));
					else
						fi->C() = GetColorForValue(r, 0.25, 1, 2);
				}
			}
		} break;

		case 5: return;

		default: break;
		}
	}

	// =====================================================================================
	// AVERAGING OVER MULTIPLE METRICS (mixColors = true means accumulation)
	// =====================================================================================
	else if (averageOfSpecific_or_UseRanges == 1 || averageOfSpecific_or_UseRanges == 2) {
		switch (metric) {
		// -------------------------
		// 0) radii
		// -------------------------
		case 0: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					CMeshO::FacePointer fp    = &*fi;
					double              ratio = CalculateCircleRadii(fi);

					if (mixColors) {
						double newData      = data_struct.map.at(fp) + ratio;
						data_struct.map[fp] = newData;
					}
					else {
						data_struct.map[fp] = ratio;
					}
				}
			}
		} break;

		// -------------------------
		// 1) area
		// -------------------------
		case 1: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				double area = 0;
				if (!fi->IsD() && (area = CalculateTriangleArea(fi)) != 0) {
					CMeshO::FacePointer fp = &*fi;

					if (mixColors) {
						double newData      = data_struct.map.at(fp) + area;
						data_struct.map[fp] = newData;
					}
					else {
						data_struct.map[fp] = area;
					}
				}
			}
		} break;

		// -------------------------
		// 2) min/max angle
		// -------------------------
		case 2: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					double              angle = CalculateMinMaxAngles(fi);
					CMeshO::FacePointer fp    = &*fi;

					if (mixColors) {
						double newData      = data_struct.map.at(fp) + angle;
						data_struct.map[fp] = newData;
					}
					else {
						data_struct.map[fp] = angle;
					}
				}
			}
		} break;

		// -------------------------
		// 3) ratio
		// -------------------------
		case 3: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					double              r  = CalculateMaxMinSideRatio(fi);
					CMeshO::FacePointer fp = &*fi;

					if (mixColors) {
						double newData      = data_struct.map.at(fp) + r;
						data_struct.map[fp] = newData;
					}
					else {
						data_struct.map[fp] = r;
					}
				}
			}
		} break;

		// -------------------------
		// 4) heights
		// -------------------------
		case 4: {
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (!fi->IsD() && CalculateTriangleArea(fi) != 0) {
					double              r  = CalculateMinMaxHeightsRatio(fi);
					CMeshO::FacePointer fp = &*fi;

					if (mixColors) {
						double newData      = data_struct.map.at(fp) + r;
						data_struct.map[fp] = newData;
					}
					else {
						data_struct.map[fp] = r;
					}
				}
			}
		} break;

		case 5: return;

		default: break;
		}
	}

	else {
		throw MLException("Bad variable value");
	}
}


double Plugin::computeAngleUsingCosineLaw(double a, double b, double c)
{
	double cosC = (a * a + b * b - c * c) / (2.0 * a * b);
	return std::acos(cosC) * (180.0 / M_PI); // Return angle in radians
}




// function whitch apply second visualization
void Plugin::FP_SECOND_VIS_Apply(CMeshO& mesh, int mixColorsMode)
{
	if (mixColorsMode == 0) { // Use average of vertex angles;

		map<CMeshO::VertexPointer, Color4b> map;

		for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
			if (!(*fi).IsD() && CalculateTriangleArea(fi) != 0) {
				CMeshO::VertexPointer v0 = (*fi).V(0);
				CMeshO::VertexPointer v1 = (*fi).V(1);
				CMeshO::VertexPointer v2 = (*fi).V(2);

				double a = CalculateDistance(v1->P(), v2->P());
				double b = CalculateDistance(v0->P(), v2->P());
				double c = CalculateDistance(v0->P(), v1->P());

				double a0 = computeAngleUsingCosineLaw(b, c, a);
				double a1 = computeAngleUsingCosineLaw(c, a, b);
				double a2 = computeAngleUsingCosineLaw(a, b, c);

				Color4b color = GetColorForValue(a0, 0, 60, 180);
				if (map.find(v0) != map.end()) {
					map[v0] = AverageColors(color, map.at(v0));
				}
				else {
					map[v0] = color;
				}

				color = GetColorForValue(a1, 0, 60, 180);
				if (map.find(v1) != map.end()) {
					map[v1] = AverageColors(color, map.at(v1));
				}
				else {
					map[v1] = color;
				}

				color = GetColorForValue(a2, 0, 60, 180);
				if (map.find(v2) != map.end()) {
					map[v2] = AverageColors(color, map.at(v2));
				}
				else {
					map[v2] = color;
				}
			}
		}

		for (CMeshO::VertexIterator vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
			if (!(*vi).IsD()) {
				try {
					(*vi).C() = map.at(&*vi);
				}
				catch (const std::exception& e) {
				}
			}
		}
	}
	else if (mixColorsMode == 1) {
		map<CMeshO::VertexPointer, double> map;

		for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
			if (!(*fi).IsD() && CalculateTriangleArea(fi) != 0) {
				CMeshO::VertexPointer v0 = (*fi).V(0);
				CMeshO::VertexPointer v1 = (*fi).V(1);
				CMeshO::VertexPointer v2 = (*fi).V(2);

				double a = CalculateDistance(v1->P(), v2->P());
				double b = CalculateDistance(v0->P(), v2->P());
				double c = CalculateDistance(v0->P(), v1->P());

				double a0 = computeAngleUsingCosineLaw(b, c, a);
				double a1 = computeAngleUsingCosineLaw(c, a, b);
				double a2 = computeAngleUsingCosineLaw(a, b, c);

				if (map.find(v0) != map.end()) {
					double angleInMap = map.at(v0);
					double dif1 = 0, dif2 = 0;
					if (angleInMap > 60) {
						dif1 = angleInMap - 60;
					}
					else {
						dif1 = 60 - angleInMap;
					}

					if (a0 > 60) {
						dif2 = a0 - 60;
					}
					else {
						dif2 = 60 - a0;
					}

					if (dif2 > dif1) {
						map[v0] = a0;
					}
				}
				else {
					map[v0] = a0;
				}

				if (map.find(v1) != map.end()) {
					double angleInMap = map.at(v1);
					double dif1 = 0, dif2 = 0;
					if (angleInMap > 60) {
						dif1 = angleInMap - 60;
					}
					else {
						dif1 = 60 - angleInMap;
					}

					if (a1 > 60) {
						dif2 = a1 - 60;
					}
					else {
						dif2 = 60 - a1;
					}

					if (dif2 > dif1) {
						map[v1] = a1;
					}
				}
				else {
					map[v1] = a1;
				}

				if (map.find(v2) != map.end()) {
					double angleInMap = map.at(v2);
					double dif1 = 0, dif2 = 0;
					if (angleInMap > 60) {
						dif1 = angleInMap - 60;
					}
					else {
						dif1 = 60 - angleInMap;
					}

					if (a2 > 60) {
						dif2 = a2 - 60;
					}
					else {
						dif2 = 60 - a2;
					}

					if (dif2 > dif1) {
						map[v2] = a2;
					}
				}
				else {
					map[v2] = a2;
				}
			}
		}

		for (CMeshO::VertexIterator vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
			if (!(*vi).IsD()) {
				try {
					(*vi).C() = GetColorForValue(map.at(&*vi), 0, 60, 180);
				}
				catch (const std::exception& e) {
				}
			}
		}
	}
	else {
		throw MLException("Bad variable value");
	}
}



static inline double safeCotFromAngle(double angleRad)
{
	// cot(x) = cos(x)/sin(x), stabilizácia pre malé sin
	double s = sin(angleRad);
	if (!std::isfinite(s) || fabs(s) < 1e-12)
		return 0.0;
	double c = cos(angleRad);
	if (!std::isfinite(c))
		return 0.0;
	return c / s;
}



double Plugin::ComputeMeanCurvature(CMeshO::VertexPointer v, CMeshO& mesh)
{
	if (!v || v->IsD())
		return 0.0;
	if (!isValidPoint(v->P()))
		return 0.0;

	// musí existovať adjacency: v -> incident faces
	auto it = m_vertFaceAdj.find(v);
	if (it == m_vertFaceAdj.end() || it->second.empty())
		return 0.0;

	// ---- DEBUG LOG  ----
	static int logCounter = 0;
	//const bool doLog      = (logCounter < 200); // zapíše len prvých 200 volaní
	const bool doLog = false;
	if (doLog)
		++logCounter;

	std::ostringstream ss;
	if (doLog) {
		const auto& p = v->P();
		ss << "---- ComputeMeanCurvature ----\n";
		ss << "vPtr=" << (void*) v << " P=(" << p.X() << "," << p.Y() << "," << p.Z() << ")\n";
		ss << "incidentFaces=" << it->second.size() << "\n";
	}

	const vcg::Point3f& pi_f = v->P();
	vcg::Point3d        pi(pi_f.X(), pi_f.Y(), pi_f.Z());

	vcg::Point3d laplace(0, 0, 0);
	double       area = 0.0;

	int usedFaces    = 0;
	int skippedFaces = 0;

	for (CMeshO::FacePointer fp : it->second) {
		if (!fp || fp->IsD()) {
			skippedFaces++;
			continue;
		}

		CMeshO::VertexPointer fv[3] = {fp->V(0), fp->V(1), fp->V(2)};
		if (!fv[0] || !fv[1] || !fv[2]) {
			skippedFaces++;
			continue;
		}
		if (fv[0]->IsD() || fv[1]->IsD() || fv[2]->IsD()) {
			skippedFaces++;
			continue;
		}
		if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0]) {
			skippedFaces++;
			continue;
		}

		// nájdi index v tejto face
		int idx = -1;
		for (int i = 0; i < 3; ++i)
			if (fv[i] == v) {
				idx = i;
				break;
			}
		if (idx < 0) {
			skippedFaces++;
			continue;
		}

		CMeshO::VertexPointer vj = fv[(idx + 1) % 3];
		CMeshO::VertexPointer vk = fv[(idx + 2) % 3];
		if (!vj || !vk) {
			skippedFaces++;
			continue;
		}
		if (!isValidPoint(vj->P()) || !isValidPoint(vk->P())) {
			skippedFaces++;
			continue;
		}

		vcg::Point3d pj(vj->P().X(), vj->P().Y(), vj->P().Z());
		vcg::Point3d pk(vk->P().X(), vk->P().Y(), vk->P().Z());

		vcg::Point3d a1    = pi - pk;
		vcg::Point3d b1    = pj - pk;
		double       ang_k = vcg::Angle(a1, b1);

		vcg::Point3d a2    = pi - pj;
		vcg::Point3d b2    = pk - pj;
		double       ang_j = vcg::Angle(a2, b2);

		if (!std::isfinite(ang_k) || !std::isfinite(ang_j)) {
			skippedFaces++;
			continue;
		}

		double cot_k = safeCotFromAngle(ang_k);
		double cot_j = safeCotFromAngle(ang_j);

		laplace += (pi - pj) * cot_k;
		laplace += (pi - pk) * cot_j;

		vcg::Point3d e1       = pj - pi;
		vcg::Point3d e2       = pk - pi;
		double       triArea2 = (e1 ^ e2).Norm();
		double       triArea  = 0.5 * triArea2;
		if (std::isfinite(triArea) && triArea > 0.0)
			area += triArea / 3.0;

		usedFaces++;

		if (doLog && usedFaces <= 5) { // vypíš len prvých pár face
			ss << "faceUsed #" << usedFaces << " ang_k=" << ang_k << " ang_j=" << ang_j
			   << " cot_k=" << cot_k << " cot_j=" << cot_j << " triArea=" << triArea << "\n";
		}
	}

	if (doLog) {
		ss << "usedFaces=" << usedFaces << " skippedFaces=" << skippedFaces << "\n";
		ss << "area=" << area << " laplace=(" << laplace.X() << "," << laplace.Y() << ","
		   << laplace.Z() << ")\n";
	}

	if (!(area > 1e-18) || !std::isfinite(area)) {
		if (doLog)
			ss << "RESULT: area too small -> return 0\n";
		if (doLog) {


			FILE* fw = fopen("C:/Users/marti/OneDrive/Desktop/curvature_debug.txt", "a");


			if (fw) {
				fputs(ss.str().c_str(), fw);
				fclose(fw);
			}
		}
		return 0.0;
	}

	laplace /= (2.0 * area);
	double H = laplace.Norm();
	if (!std::isfinite(H))
		H = 0.0;

	if (doLog) { 
		ss << "H=" << H << "\n\n";
		FILE* fw = fopen("C:/Users/marti/OneDrive/Desktop/curvature_debug.txt", "a");
		if (fw) {
			fputs(ss.str().c_str(), fw);
			fclose(fw);
		}
	} 

	return H;
}

AdjMap BuildVertexAdjacency(CMeshO& mesh)
{
	AdjMap adj;
	adj.reserve((size_t) mesh.vn);

	for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
		if (fi->IsD())
			continue;

		VertexPtr v[3];
		for (int i = 0; i < 3; ++i) {
			v[i] = fi->V(i);
			if (!v[i] || v[i]->IsD()) {
				v[0] = nullptr;
				break;
			}
		}
		if (!v[0])
			continue;

		if (v[0] == v[1] || v[1] == v[2] || v[2] == v[0])
			continue;

		for (int i = 0; i < 3; ++i) {
			VertexPtr a = v[i];
			VertexPtr b = v[(i + 1) % 3];
			adj[a].push_back(b);
			adj[b].push_back(a);
		}
	}
	return adj;
}






QualityMap
ComputeNeighborhoodScore(CMeshO& mesh, const QualityMap& baseScore, const AdjMap& adj, int radius)
{
	QualityMap neighScore;
	neighScore.reserve(baseScore.size());

	if (radius <= 0)
		return baseScore;

	std::queue<std::pair<VertexPtr, int>> q;
	std::unordered_set<VertexPtr>         visited;
	visited.reserve(256);

	for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
		if (vi->IsD())
			continue;
		VertexPtr v0 = &*vi;

		auto it0 = baseScore.find(v0);
		if (it0 == baseScore.end())
			continue;

		while (!q.empty())
			q.pop();
		visited.clear();

		visited.insert(v0);
		q.push({v0, 0});

		// zahrň aj seba
		double sum = it0->second;
		int    cnt = 1;

		while (!q.empty()) {
			VertexPtr v    = q.front().first;
			int       dist = q.front().second;
			q.pop();

			if (dist == radius)
				continue;

			auto itAdj = adj.find(v);
			if (itAdj == adj.end())
				continue;

			for (VertexPtr n : itAdj->second) {
				if (!n)
					continue;
				if (visited.find(n) != visited.end())
					continue;

				visited.insert(n);
				q.push({n, dist + 1});

				auto itN = baseScore.find(n);
				if (itN != baseScore.end()) {
					sum += itN->second;
					cnt++;
				}
			}
		}

		neighScore[v0] = (cnt > 0) ? (sum / (double) cnt) : it0->second;
	}

	return neighScore;
}



QualityMap ApplyNeighborhoodPostprocessing(
	const QualityMap& baseScore,
	const QualityMap& neighScore,
	NeighborhoodMode  mode,
	double            weight,
	bool              higherIsBetter,
	bool              useOptimalityBased)
{
	QualityMap out;
	out.reserve(baseScore.size());

	for (auto& kv : baseScore) {
		VertexPtr v       = kv.first;
		double    baseVal = kv.second;

		auto   itN      = neighScore.find(v);
		double neighVal = (itN != neighScore.end()) ? itN->second : baseVal;

		if (mode == NEIGH_OFF || weight <= 0.0) {
			out[v] = baseVal;
			continue;
		}

		double blended = (1.0 - weight) * baseVal + weight * neighVal;

		if (mode == NEIGH_SMOOTH) {
			out[v] = blended;
		}
		else { // NEIGH_IMPROVE_ONLY
			if (useOptimalityBased) {
				// lepšie = bližšie k 0.5
				double distBase    = std::fabs(baseVal - 0.5);
				double distBlended = std::fabs(blended - 0.5);

				if (distBlended < distBase)
					out[v] = blended;
				else
					out[v] = baseVal;
			}
			else {
				// pôvodná logika pre range-based
				if (higherIsBetter)
					out[v] = std::max(baseVal, blended);
				else
					out[v] = std::min(baseVal, blended);
			}
		}
	}

	return out;
}




// main filter function
std::map<std::string, QVariant> Plugin::applyFilter(
	const QAction*           filter,
	const RichParameterList& par,
	MeshDocument&            md,
	unsigned int& /*postConditionMask*/,
	vcg::CallBackPos*)


{

	//auto start = high_resolution_clock::now();

	MeshModel* m = md.mm();
	if (!m)
		return std::map<std::string, QVariant>();

	if (ID(filter) == FP_FIRST_VIS) {
		m->updateDataMask(MeshModel::MM_FACEFACETOPO);
		m->updateDataMask(MeshModel::MM_FACEMARK | MeshModel::MM_FACECOLOR);

		CMeshO& mesh = m->cm;
		RequirePerFaceColor(mesh);

		if (!AreAllFacesTriangles(mesh)) {
			throw MLException("The mesh must contain only triangles");
			return std::map<std::string, QVariant>();
		}

		// init data_struct
		data_struct.counter = 0;
		data_struct.max     = 0;
		data_struct.min     = DBL_MAX;
		data_struct.sum     = 0;
		data_struct.map.clear();

		// run 1–5 metrics in order
		int colorMixingMode = par.getEnum("colorMixFaktor");

		int metric = par.getEnum("metric1");
		FP_FIRST_VIS_Apply(mesh, metric, colorMixingMode, false);

		metric = par.getEnum("metric2");
		FP_FIRST_VIS_Apply(mesh, metric, colorMixingMode, true);

		metric = par.getEnum("metric3");
		FP_FIRST_VIS_Apply(mesh, metric, colorMixingMode, true);

		metric = par.getEnum("metric4");
		FP_FIRST_VIS_Apply(mesh, metric, colorMixingMode, true);

		metric = par.getEnum("metric5");
		FP_FIRST_VIS_Apply(mesh, metric, colorMixingMode, true);

		// -------------------------------------------
		//  FINAL COLORING (only for mode 1 and 2)
		// -------------------------------------------
		if (colorMixingMode == 1 || colorMixingMode == 2) {
			// recompute min/max/sum on final map values
			for (auto& kv : data_struct.map) {
				double val = kv.second;
				data_struct.sum += val;
				data_struct.max = std::max(data_struct.max, val);
				data_struct.min = std::min(data_struct.min, val);
				data_struct.counter++;
			}

			// apply face colors using corrected map keys
			for (CMeshO::FaceIterator fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
				if (fi->IsD())
					continue;

				CMeshO::FacePointer fp = &*fi;

				if (data_struct.map.find(fp) == data_struct.map.end())
					continue;

				double val = data_struct.map.at(fp);

				try {
					switch (colorMixingMode) {
					case 1: // optimal = mean
						fi->C() = GetColorForValue(
							val,
							data_struct.min,
							data_struct.sum / data_struct.counter,
							data_struct.max);
						break;

					case 2: // optimal = midpoint of min/max
						fi->C() = GetColorForValue(
							val,
							data_struct.min,
							(data_struct.min + data_struct.max) / 2,
							data_struct.max);
						break;

					default: break;
					}
				}
				catch (...) {
					// swallow exceptions just like before
				}
			}
		}

		// reset data_struct
		data_struct.sum     = 0;
		data_struct.counter = 0;
		data_struct.min     = DBL_MAX;
		data_struct.max     = DBL_MIN;
		data_struct.map.clear();
	}


	else if (ID(filter) == FP_SECOND_VIS) {
		MeshModel* m = md.mm();
		m->updateDataMask(MeshModel::MM_VERTCOLOR);
		CMeshO& mesh = m->cm;
		RequirePerVertexColor(mesh);

		if (!AreAllFacesTriangles(mesh)) {
			throw MLException("The mesh must contain only triangles");
			return std::map<std::string, QVariant>();
		}
		int colorMixingMode = par.getEnum("colorMixFaktor1");
		FP_SECOND_VIS_Apply(mesh, colorMixingMode);
	}

// tretia vizualizácia - aktualne rozpracovaná pre dipl prácu//
	else if (ID(filter) == FP_Third_VIS) {
		// =========================
		// Logging setup
		// =========================
		ThirdVisLogger L;
		L.enabled = par.getBool("enableLogging");
		L.mode    = (ThirdVisLogger::Mode) par.getEnum("logVerbosity");

		const bool doOverview  = L.enabled && (L.mode == ThirdVisLogger::Mode::Overview);
		const bool doPerVertex = L.enabled && (L.mode == ThirdVisLogger::Mode::PerVertex);

		// optional histogram image export
		const bool saveHistogram = par.getBool("saveHistogramImage");

		// Overview buffer (MeshLab log)
		std::ostringstream overview;
		if (doOverview) {
			overview << "================ FP_Third_VIS ================\n";
			overview << "Logging: Overview -> MeshLab Log\n";
		}

		// =========================
		// Mesh init
		// =========================
		MeshModel* m = md.mm();
		m->updateDataMask(MeshModel::MM_VERTCOLOR);
		CMeshO& mesh = m->cm;
		RequirePerVertexColor(mesh);

		if (!AreAllFacesTriangles(mesh)) {
			throw MLException("The mesh must contain only triangles");
			return std::map<std::string, QVariant>();
		}

		BuildVertexFaceAdjacency(mesh);

		int   metricA           = par.getEnum("vertexMetricA");
		int   metricB           = par.getEnum("vertexMetricB");
		float mixRatio          = par.getFloat("metricMixRatio");
		int   normalizationMode = par.getEnum("normalizationMode");

		if (metricA == 0 && metricB != 0)
			std::swap(metricA, metricB);

		// Per-vertex -> file
		if (doPerVertex) {
			L.filePath = par.getString("logOutputFile");
			if (L.filePath.trimmed().isEmpty()) {
				L.enabled = false;
			}
			else {
				L.line("================ FP_Third_VIS ================");
				L.line("Logging: Per-vertex -> File");
				L.line(std::string("File: ") + L.filePath.toStdString());
				L.line(
					std::string("Normalization mode: ") +
					(normalizationMode == 0 ? "Range-based" : "Optimality-based"));

				if (normalizationMode == 1) {
					double optAHeader = GetMetricOptimalValue(metricA);
					double optAMapped = 0.0;

					// transform helper lokálne len pre header
					if (metricA == 7)
						optAMapped = std::log1p(std::max(0.0, optAHeader));
					else if (metricA == 4 || metricA == 6)
						optAMapped = std::log1p(std::max(0.0, optAHeader));
					else
						optAMapped = optAHeader;

					L.line(std::string("A optimum (mapped): ") + std::to_string(optAMapped));

					if (metricB != 0) {
						double optBHeader = GetMetricOptimalValue(metricB);
						double optBMapped = 0.0;

						if (metricB == 7)
							optBMapped = std::log1p(std::max(0.0, optBHeader));
						else if (metricB == 4 || metricB == 6)
							optBMapped = std::log1p(std::max(0.0, optBHeader));
						else
							optBMapped = optBHeader;

						L.line(std::string("B optimum (mapped): ") + std::to_string(optBMapped));
					}
				}

				L.line("------------------------------------------------");
			}
		}

		if (doOverview) {
			overview << "mesh.vn=" << mesh.vn << "  mesh.fn=" << mesh.fn << "\n";
			overview << "metricA=" << metricA << "  metricB=" << metricB
					 << "  mixRatio=" << mixRatio << "\n";
			overview << "normalizationMode="
					 << (normalizationMode == 0 ? "Range-based" : "Optimality-based") << "\n";
		}

		if (metricA == 0 && metricB == 0) {
			for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
				if (!vi->IsD())
					vi->C() = vcg::Color4b(0, 255, 0, 255);
			}

			if (doOverview) {
				overview << "\nA=None, B=None => neutral coloring\n";
				LogMultiline(md.Log, GLLogStream::FILTER, overview.str());
			}
			return std::map<std::string, QVariant>();
		}

		using VertexPtr = CMeshO::VertexPointer;

		// =========================
		// Stats containers (for Overview)
		// =========================
		RunningStats rawAStats, rawBStats;
		RunningStats mapAStats, mapBStats;
		RunningStats scorePreStats, scorePostStats;

		// cache pre curvature
		std::unordered_map<VertexPtr, double> curvatureCache;
		curvatureCache.reserve((size_t) std::max(1, mesh.vn));

		auto getCurvatureCached = [&](VertexPtr v) -> double {
			auto it = curvatureCache.find(v);
			if (it != curvatureCache.end())
				return it->second;

			double c = ComputeMeanCurvature(v, mesh);
			if (!isValidNumber(c) || !IsFinite(c) || c < 0.0)
				c = 0.0;

			curvatureCache.emplace(v, c);
			return c;
		};

		auto computeMetricCached = [&](int metricID, VertexPtr v) -> double {
			if (metricID == 0)
				return 0.0;
			if (metricID == 7)
				return getCurvatureCached(v);

			double x = ComputeVertexMetric(metricID, v, mesh);
			if (!isValidNumber(x) || !IsFinite(x))
				x = 0.0;
			return x;
		};

		auto transformForMapping = [&](int metricID, double raw) -> double {
			if (!IsFinite(raw) || !isValidNumber(raw))
				return 0.0;

			if (metricID == 7) // MeanCurvature
				return std::log1p(std::max(0.0, raw));

			if (metricID == 4 /*EdgeLengthRatio*/ || metricID == 6 /*AngleDeviation360*/)
				return std::log1p(std::max(0.0, raw));

			return raw;
		};

		// ---- hodnoty pre všetky vertexy ----
		std::unordered_map<VertexPtr, double> rawAmap, rawBmap;
		rawAmap.reserve((size_t) mesh.vn);
		rawBmap.reserve((size_t) mesh.vn);

		std::vector<double> Avals, Bvals;
		Avals.reserve((size_t) mesh.vn);
		Bvals.reserve((size_t) mesh.vn);

		int processedVerts = 0;
		int skippedDeleted = 0;

		const bool sameMetric = (metricB != 0 && metricA == metricB);

		// 1) zber raw + mapped
		for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
			if (vi->IsD()) {
				skippedDeleted++;
				continue;
			}
			processedVerts++;
			VertexPtr v = &*vi;

			double aRaw = computeMetricCached(metricA, v);
			rawAmap[v]  = aRaw;
			rawAStats.add(aRaw);

			double aT = transformForMapping(metricA, aRaw);
			if (IsFinite(aT)) {
				Avals.push_back(aT);
				mapAStats.add(aT);
			}

			if (metricB != 0 && !sameMetric) {
				double bRaw = computeMetricCached(metricB, v);
				rawBmap[v]  = bRaw;
				rawBStats.add(bRaw);

				double bT = transformForMapping(metricB, bRaw);
				if (IsFinite(bT)) {
					Bvals.push_back(bT);
					mapBStats.add(bT);
				}
			}
		}

		if (sameMetric) {
			rawBmap   = rawAmap;
			Bvals     = Avals;
			rawBStats = rawAStats;
			mapBStats = mapAStats;
		}

		if (doOverview) {
			overview << "processedVerts=" << processedVerts
					 << "  skippedDeletedVerts=" << skippedDeleted << "\n";
		}

		// 2) robust range (percentiles)
		const double P_LO = 0.01;
		const double P_HI = 0.99;

		double minA = 0.0, maxA = 1.0;
		double minB = 0.0, maxB = 1.0;

		// A robust
		if (!Avals.empty()) {
			std::vector<double> tmp = Avals;
			double              pLo = Percentile(tmp, P_LO);
			double              pHi = Percentile(tmp, P_HI);
			if (IsFinite(pLo) && IsFinite(pHi) && pHi > pLo) {
				minA = pLo;
				maxA = pHi;
			}
			else {
				auto mm = std::minmax_element(Avals.begin(), Avals.end());
				minA    = *mm.first;
				maxA    = *mm.second;
				if (!(maxA > minA)) {
					minA = 0.0;
					maxA = 1.0;
				}
			}
		}

		// B robust
		if (metricB != 0 && !Bvals.empty()) {
			std::vector<double> tmp = Bvals;
			double              pLo = Percentile(tmp, P_LO);
			double              pHi = Percentile(tmp, P_HI);
			if (IsFinite(pLo) && IsFinite(pHi) && pHi > pLo) {
				minB = pLo;
				maxB = pHi;
			}
			else {
				auto mm = std::minmax_element(Bvals.begin(), Bvals.end());
				minB    = *mm.first;
				maxB    = *mm.second;
				if (!(maxB > minB)) {
					metricB = 0;
					rawBmap.clear();
				}
			}
		}

		if (doOverview) {
			overview << "\n--- Metric stats ---\n";
			AppendStats(overview, "A raw   ", rawAStats);
			AppendStats(overview, "A mapped", mapAStats);
			overview.setf(std::ios::fixed);
			overview.precision(6);
			overview << "A robust : loP=" << P_LO << " hiP=" << P_HI << "  minA=" << minA
					 << "  maxA=" << maxA << "\n";

			if (metricB != 0) {
				overview << "\n";
				AppendStats(overview, "B raw   ", rawBStats);
				AppendStats(overview, "B mapped", mapBStats);
				overview << "B robust : loP=" << P_LO << " hiP=" << P_HI << "  minB=" << minB
						 << "  maxB=" << maxB << "\n";
			}
			else {
				overview << "\nB: None\n";
			}
		}

		if (doOverview && normalizationMode == 1) {
			overview << "\n--- Optimality-based normalization ---\n";

			double optA = transformForMapping(metricA, GetMetricOptimalValue(metricA));
			overview << "A optimum (mapped)=" << optA << "\n";

			if (metricB != 0) {
				double optB = transformForMapping(metricB, GetMetricOptimalValue(metricB));
				overview << "B optimum (mapped)=" << optB << "\n";
			}
		}

		// 3) scoreMap (pre-neigh)
		double safeMix = mixRatio;
		if (!isValidNumber(safeMix) || !IsFinite(safeMix))
			safeMix = 0.5;
		safeMix = (double) Clamp(safeMix, 0.0, 1.0);

		std::unordered_map<VertexPtr, double> scoreMap;
		scoreMap.reserve((size_t) mesh.vn);

		auto norm01 = [&](double x, double lo, double hi) -> double {
			double denom = hi - lo;
			if (!IsFinite(x) || !IsFinite(lo) || !IsFinite(hi) || !(denom > 1e-12))
				return 0.5;
			double t = (x - lo) / denom;
			if (!IsFinite(t))
				return 0.5;
			return Clamp(t, 0.0, 1.0);
		};

		auto normAroundOptimal = [&](double x, double opt, double lo, double hi) -> double {
			if (!IsFinite(x) || !IsFinite(opt) || !IsFinite(lo) || !IsFinite(hi))
				return 0.5;

			if (opt < lo || opt > hi)
				return 0.5;

			double leftRange  = opt - lo;
			double rightRange = hi - opt;

			if (x <= opt) {
				if (leftRange <= 1e-12)
					return 0.5;

				double t = (opt - x) / leftRange;
				if (!IsFinite(t))
					return 0.5;

				return Clamp(0.5 - 0.5 * t, 0.0, 0.5);
			}
			else {
				if (rightRange <= 1e-12)
					return 0.5;

				double t = (x - opt) / rightRange;
				if (!IsFinite(t))
					return 0.5;

				return Clamp(0.5 + 0.5 * t, 0.5, 1.0);
			}
		};

		double minScore = std::numeric_limits<double>::infinity();
		double maxScore = -std::numeric_limits<double>::infinity();

		for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
			if (vi->IsD())
				continue;
			VertexPtr v = &*vi;

			double aRaw = 0.0;
			auto   itA  = rawAmap.find(v);
			if (itA != rawAmap.end())
				aRaw = itA->second;

			double aT    = transformForMapping(metricA, aRaw);
			double normA = 0.5;

			if (normalizationMode == 0) {
				normA = norm01(aT, minA, maxA);
			}
			else {
				double optA = transformForMapping(metricA, GetMetricOptimalValue(metricA));
				normA       = normAroundOptimal(aT, optA, minA, maxA);
			}

			double score = normA;

			if (metricB != 0) {
				double bRaw = 0.0;
				auto   itB  = rawBmap.find(v);
				if (itB != rawBmap.end())
					bRaw = itB->second;

				double bT    = transformForMapping(metricB, bRaw);
				double normB = 0.5;

				if (normalizationMode == 0) {
					normB = norm01(bT, minB, maxB);
				}
				else {
					double optB = transformForMapping(metricB, GetMetricOptimalValue(metricB));
					normB       = normAroundOptimal(bT, optB, minB, maxB);
				}

				score = safeMix * normA + (1.0 - safeMix) * normB;
			}

			if (!IsFinite(score) || !isValidNumber(score))
				score = 0.5;

			score = Clamp(score, 0.0, 1.0);

			scoreMap[v] = score;
			scorePreStats.add(score);

			minScore = std::min(minScore, score);
			maxScore = std::max(maxScore, score);
		}

		if (doOverview) {
			overview << "\n--- Score stats ---\n";
			AppendStats(overview, "score pre", scorePreStats);
			overview.setf(std::ios::fixed);
			overview.precision(6);
			overview << "spread    : " << (maxScore - minScore) << "  min=" << minScore
					 << "  max=" << maxScore << "\n";
		}

		// neighborhood params
		int radius = par.getInt("neighborhoodRadius");
		if (radius < 0)
			radius = 0;

		int    modeInt = par.getEnum("neighborhoodMode");
		double weight  = par.getFloat("neighborhoodWeight");
		if (weight < 0.0)
			weight = 0.0;
		if (weight > 1.0)
			weight = 1.0;

		NeighborhoodMode neighMode = static_cast<NeighborhoodMode>(modeInt);

		bool neighApplied   = false;
		bool higherIsBetter = true;

		if (doOverview) {
			overview << "\n--- Neighborhood ---\n";
			overview.setf(std::ios::fixed);
			overview.precision(6);
			if (radius == 0 || neighMode == NEIGH_OFF || weight <= 0.0) {
				overview << "Neighborhood: Off\n";
			}
			else {
				overview << "radius=" << radius << "  mode=" << modeInt << "  weight=" << weight
						 << "\n";
			}
		}

		if (radius > 0 && neighMode != NEIGH_OFF && weight > 0.0) {
			neighApplied = true;

			auto       adj        = BuildVertexAdjacency(mesh);
			QualityMap neighScore = ComputeNeighborhoodScore(mesh, scoreMap, adj, radius);

			higherIsBetter = true;
			if (metricA == 6 /*AngleDeviation360*/ || metricB == 6 /*AngleDeviation360*/)
				higherIsBetter = false;

			QualityMap finalScore = ApplyNeighborhoodPostprocessing(
				scoreMap, neighScore, neighMode, weight, higherIsBetter, normalizationMode == 1);

			scoreMap.swap(finalScore);

			if (doOverview) {
				scorePostStats = RunningStats {};
				for (auto& kv : scoreMap) {
					double s = kv.second;
					if (!IsFinite(s) || !isValidNumber(s))
						s = 0.5;
					s = Clamp(s, 0.0, 1.0);
					scorePostStats.add(s);
				}

				overview << "neigh applied: yes  higherIsBetter=" << (higherIsBetter ? 1 : 0)
						 << "\n";
				AppendStats(overview, "score post", scorePostStats);
			}
		}
		else {
			if (doOverview) {
				overview << "neigh applied: no\n";
			}
		}

		// =========================
		// Per-vertex dump (AFTER neigh, BEFORE coloring)
		// =========================
		if (doPerVertex && L.enabled) {
			L.line("idx x y z rawA rawB score");

			int idx = 0;
			for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
				if (vi->IsD())
					continue;

				VertexPtr   v = &*vi;
				const auto& p = v->P();

				double aRaw = 0.0, bRaw = 0.0, s = 0.5;

				auto itA = rawAmap.find(v);
				if (itA != rawAmap.end())
					aRaw = itA->second;

				if (metricB != 0) {
					auto itB = rawBmap.find(v);
					if (itB != rawBmap.end())
						bRaw = itB->second;
				}

				auto itS = scoreMap.find(v);
				if (itS != scoreMap.end())
					s = itS->second;

				std::ostringstream row;
				row.setf(std::ios::fixed);
				row.precision(6);
				row << idx << " " << p.X() << " " << p.Y() << " " << p.Z() << " " << aRaw << " "
					<< bRaw << " " << s;

				L.line(row.str());
				idx++;
			}
		}

		// 4) Coloring
		const double opt = 0.5;

		int coloredVerts = 0;
		for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
			if (vi->IsD())
				continue;

			VertexPtr v = &*vi;

			double s   = 0.5;
			auto   itS = scoreMap.find(v);
			if (itS != scoreMap.end() && IsFinite(itS->second) && isValidNumber(itS->second))
				s = itS->second;

			vi->C() = GetColorForValue(s, 0.0, opt, 1.0);
			coloredVerts++;
		}

		if (doOverview) {
			overview << "\n--- Coloring ---\n";
			overview << "coloredVerts=" << coloredVerts << " (expected ~ " << mesh.vn << ")\n";
		}

		// =========================
		// Optional histogram export
		// =========================
		if (L.enabled && doPerVertex && saveHistogram && !L.filePath.trimmed().isEmpty()) {
			std::vector<double> histValues;
			histValues.reserve(scoreMap.size());

			for (auto& kv : scoreMap) {
				double s = kv.second;
				if (!IsFinite(s) || !isValidNumber(s))
					s = 0.5;
				s = Clamp(s, 0.0, 1.0);
				histValues.push_back(s);
			}

			std::vector<float> histBins = BuildHistogramBins(histValues, 30);

			QString histPath = makeHistogramFilePath(L.filePath);
			saveHistogramImage(histBins, histPath.toStdString());

			L.line(std::string("Histogram image: ") + histPath.toStdString());
		}

		// =========================
		// Flush logs
		// =========================
		if (L.enabled) {
			if (L.mode == ThirdVisLogger::Mode::Overview) {
				LogMultiline(md.Log, GLLogStream::FILTER, overview.str());
			}
			else {
				if (!L.flushToFileAppend()) {
					md.Log.log(
						GLLogStream::WARNING, "FP_Third_VIS: failed to write per-vertex log file.");
				}
				L.clear();
			}
		}

		return std::map<std::string, QVariant>();
	}


	return std::map<std::string, QVariant>();
}

int Plugin::postCondition(const QAction* filter) const
{
	switch (ID(filter)) {
	case FP_FIRST_VIS: return MeshModel::MM_FACECOLOR | MeshModel::MM_FACEQUALITY;
	case FP_SECOND_VIS: return MeshModel::MM_VERTCOLOR;
	case FP_Third_VIS: return MeshModel::MM_VERTCOLOR;
	default: assert(0);
	}
	return MeshModel::MM_NONE;
}

int Plugin::getRequirements(const QAction*){
	return MeshModel::MM_NONE;
}

int Plugin::getPreConditions(const QAction* filter) const
{
	switch (ID(filter)) {
	case FP_FIRST_VIS: return MeshModel::MM_NONE;
	case FP_SECOND_VIS: return MeshModel::MM_NONE;
	case FP_Third_VIS: return MeshModel::MM_NONE;
	default: break;
	} 
	return MeshModel::MM_NONE;
}

MESHLAB_PLUGIN_NAME_EXPORTER(Plugin)

