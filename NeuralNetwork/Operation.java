package NeuralNetwork;

public enum Operation {
    PLUS {
        @Override
        public double getValue() {
            return 0.0;
        }

        @Override
        public int calc(final int i1, final int i2) {
            return i1 + i2;
        }

        @Override
        public String toString() {
            return "+";
        }
    },
    MINUS {
        @Override
        public double getValue() {
            return 1.0;
        }

        @Override
        public int calc(final int i1, final int i2) {
            return i1 - i2;
        }

        @Override
        public String toString() {
            return "-";
        }
    },
    MULTIPLY {
        @Override
        public double getValue() {
            return 2.0;
        }

        @Override
        public int calc(final int i1, final int i2) {
            return i1 * i2;
        }

        @Override
        public String toString() {
            return "*";
        }
    },
    DIVIDE {
        @Override
        public double getValue() {
            return 3.0;
        }

        @Override
        public int calc(final int i1, final int i2) {
            return i1 / i2;
        }

        @Override
        public String toString() {
            return "/";
        }
    };

    public abstract double getValue();

    public abstract int calc(int i1, int i2);

    public abstract String toString();
}