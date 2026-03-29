#ifndef FILTER_PLUGIN_H
#define FILTER_PLUGIN_H


#include <common/plugins/interfaces/filter_plugin.h>
//#include "../edit_quality/common/transferfunction.h"
#include <vcg/complex/algorithms/stat.h> 
//#include "../edit_quality/common/meshmethods.h"
#include <QObject>
#include <map>

using namespace vcg;
using namespace std;

using VertexPtr  = CMeshO::VertexPointer;
using QualityMap = std::unordered_map<VertexPtr, double>;
using AdjMap     = std::unordered_map<VertexPtr, std::vector<VertexPtr>>;
enum NeighborhoodMode { NEIGH_OFF = 0, NEIGH_SMOOTH = 1, NEIGH_IMPROVE_ONLY = 2 };



class Plugin : public QObject, public FilterPlugin {
	Q_OBJECT
	MESHLAB_PLUGIN_IID_EXPORTER(FILTER_PLUGIN_IID)
	Q_INTERFACES(FilterPlugin)


private:
	bool AreAllFacesTriangles(CMeshO& mesh);
	double CalculateDistance(const vcg::Point3f& p1, const vcg::Point3f& p2);
	bool isNegativeNaN(double x);
	Color4b InterpolateColor(const vcg::Color4b& colorStart, const vcg::Color4b& colorEnd, double factor);
	Color4b GetColorForValue(double value, double min, double optimal, double max);
	double ComputeMeanCurvature(CMeshO::VertexPointer v, CMeshO& mesh);
	double  ComputeVertexMetric(int metricID, CMeshO::VertexPointer v, CMeshO& mesh);
	bool  Plugin::isValidNumber(double x);
	bool  Plugin::isValidPoint(const vcg::Point3f& p);

	using VertFaceAdj = std::map<CMeshO::VertexPointer, std::vector<CMeshO::FacePointer>>;
	VertFaceAdj m_vertFaceAdj;
	void  Plugin::BuildVertexFaceAdjacency(CMeshO& mesh);


	float roundUpToNice(float val);
	std::vector<float> BuildHistogramBins(const std::vector<double>& values, int binCount);
	void saveHistogramImage(const std::vector<float>& data, const std::string& filename);
	QString makeHistogramFilePath(const QString& logFilePath);
	
	double GetMetricOptimalValue(int metricID);
	double Clamp(double x, double lo, double hi);
	bool   IsFinite(double x);
	double Percentile(std::vector<double>& a, double p);

	
	double safeCotFromAngle(double angleRad);

	AdjMap BuildVertexAdjacency(CMeshO& mesh);
	QualityMap ComputeNeighborhoodScore( CMeshO& mesh, const QualityMap& baseScore, const AdjMap& adj, int radius);

	QualityMap ApplyNeighborhoodPostprocessing(
		const QualityMap& baseScore, const QualityMap& neighScore,
		NeighborhoodMode  mode, double weight,
		bool higherIsBetter, bool useOptimalityBased);

public:

    enum{
		FP_VIS,
    };

    Plugin(); 
    QString pluginName() const;  
    virtual QString filterName(ActionIDType filter) const; 
    QString pythonFilterName(ActionIDType f) const; 
    virtual QString filterInfo(ActionIDType filter) const;  
	virtual RichParameterList
	initParameterList(const QAction* a, const MeshDocument& md); 
    std::map<std::string, QVariant> applyFilter(
			const QAction* action,
			const RichParameterList & parameters,
			MeshDocument &md,
			unsigned int& postConditionMask,
			vcg::CallBackPos * cb);  
    
	FilterArity filterArity(const QAction *act) const
	{
		return FilterPlugin::SINGLE_MESH;
	}                                                           
	virtual FilterClass getClass(const QAction *a) const;  
	QString filterScriptFunctionName(ActionIDType filterID); 
	virtual int getRequirements(const QAction* filter);

	virtual int getPreConditions(const QAction *filter) const;
	virtual int postCondition(const QAction *filter) const;
	

};

#endif // FILTER_PLUGIN_H
