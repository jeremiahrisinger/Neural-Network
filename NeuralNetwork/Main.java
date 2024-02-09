package NeuralNetwork;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Main extends BaseObject {

    private static final String ANSWERS = "ANSWERS";

    private static final String INPUT = "INPUT";

    private static final Logger<Main> LOGGER = new Logger<Main>(Main.class);

    private static final double[][] inputs = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 2, 0 }, { 0, 2 }, { 2, 1 },
            { 2, 2 }, { 3, 0 }, { 0, 3 }, { 1, 3 }, { 3, 1 }, { 2, 3 }, { 2, 4 }, { 4, 0 }, { 4, 1 }, { 5, 5 },
            { 5, 3 }, { 4, 3 }, { 3, 4 }, { 4, 2 }, { 5, 2 }, { 5, 1 }, { 5, 2 }, { 5, 4 }, { 4, 4 }, { 5, 0 },
            { 4, 2 }, { 4, 0 }, { 0, 5 }, { 1, 5 }, { 3, 5 }, { 2, 5 }, { 4, 5 } };

    public static void main(String[] args) {
        LOGGER.logMethod("main");

        double[][] answers = calculateAnswers(inputs);
        HashMap<String, double[][]> add = new HashMap<>();
        add.put(INPUT, inputs);
        add.put(ANSWERS, answers);

        NeuralNerwork nn = new NeuralNerwork(2, 30, 11);
        nn.fit(add, 6000000);

        List<Double> output = new ArrayList<>();

        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            try {
                output = nn.predict(input);
            } catch (Exception e) {
                LOGGER.error("Error occured while trying to predict output.", e);
            }

            LOGGER.info(String.format("input {%s, %s} -> {%s}\t%s", input[0], input[1], getAnswer(output),
                    getAnswer(output) - (input[0] + input[1])));
        }

        double[] input = { Math.round(Math.random()*5), Math.round(Math.random()*5) };
        try {
            output = nn.predict(input);
        } catch (Exception e) {
            LOGGER.error("Error occured while trying to predict output.", e);
        }

        LOGGER.info(String.format("input {%s, %s} -> {%s}\t%s", input[0], input[1], getAnswer(output),
                getAnswer(output) - (input[0] + input[1])));
    }

    private static double[][] calculateAnswers(double[][] input) {
        double[][] answers = new double[inputs.length][11];
        for (int i = 0; i < input.length; i++) {
            int sum = 0;
            for (int j = 0; j < input[i].length; j++) {
                sum += input[i][j];
            }
            answers[i][sum] = 1;
        }
        return answers;
    }

    private static int getAnswer(List<Double> output) {
        for (int i = 0; i < output.size(); i++) {
            if (Math.round(output.get(i)) == 1L)
                return i;
        }
        return -1;
    }

}
