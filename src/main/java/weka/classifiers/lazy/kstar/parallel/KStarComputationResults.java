package weka.classifiers.lazy.kstar.parallel;

/**
 * Created by I824612 on 12/10/2015.
 */
public class KStarComputationResults {
    public double[] classProbability;
    public double[] predictedValue;
    public double temp;

    public KStarComputationResults(int numberOfClasses) {
        classProbability = new double[numberOfClasses];
        predictedValue = new double[1];
        temp = 0.0;

        for (int i = 0; i < classProbability.length; i++) {
            classProbability[i] = 0.0;
        }
    }
}
