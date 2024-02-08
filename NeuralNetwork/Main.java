package NeuralNetwork;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main extends BaseObject {

    private static final String ANSWERS = "ANSWERS";

    private static final String INPUT = "INPUT";

    private static final Logger<Main> LOGGER = new Logger<Main>(Main.class);

    private static final double[][] input = {{ 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }};
    private static final double[][] orAnswers  = {{ 0 }, { 1 }, { 1 }, { 1 }};
    private static final double[][] xorAnswers = {{ 0 }, { 1 }, { 1 }, { 0 }};
    private static final double[][] andAnswers = {{ 0 }, { 0 }, { 0 }, { 1 }};
    private static final double[][] xndAnswers = {{ 1 }, { 1 }, { 1 }, { 0 }};

    public static void main(String[] args) {
        LOGGER.logMethod("main");

        HashMap<String,double[][]> and = new HashMap<>();
        and.put(INPUT, input);
        and.put(ANSWERS, andAnswers);

        HashMap<String,double[][]> or = new HashMap<>();
        or.put(INPUT, input);
        or.put(ANSWERS, orAnswers);

        HashMap<String,double[][]> xor = new HashMap<>();
        xor.put(INPUT, input);
        xor.put(ANSWERS, xorAnswers);

        HashMap<String,double[][]> xnd = new HashMap<>();
        xnd.put(INPUT, input);
        xnd.put(ANSWERS, xndAnswers);
        

        NeuralNerwork nn = new NeuralNerwork(2, 10, 1);
        nn.fit(or, 30000);
        //nn.fit(xor, 30000);
        //nn.fit(and, 30000);
        //nn.fit(xnd, 30000);


        double[][] inputs = { { 0, 1 }, { 0, 0 }, { 1, 1 }, { 1, 0 } };
        List<Double> output = new ArrayList<>();

        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            try {
                output = nn.predict(input);
            } catch (Exception e) {
                LOGGER.error("Error occured while trying to predict output.", e);
            }

            LOGGER.info(String.format("input {%s, %s} -> %s [%s]", input[0], input[1], Math.round(output.get(0)),
                    output.get(0)));

        }

    }

}
