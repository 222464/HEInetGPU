constant float epsilon = 0.00001f;

float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

void kernel rscInitialize(write_only image3d_t hiddenVisibleWeights,
	write_only image3d_t hiddenHiddenPrevWeights,
	write_only image3d_t hiddenHiddenWeights,
	int receptiveSize, int recurrentSize, int inhibitionSize, uint2 seed)
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	for (int wi = 0; wi < receptiveSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue);

		write_imagef(hiddenVisibleWeights, weightPosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
	}

	for (int wi = 0; wi < recurrentSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue);

		write_imagef(hiddenHiddenPrevWeights, weightPosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
	}

	for (int wi = 0; wi < inhibitionSize; wi++) {
		int4 weightPosition = (int4)(position.x, position.y, wi, 0);

		float weight = randFloat(&seedValue);

		write_imagef(hiddenHiddenWeights, weightPosition, (float4)(weight, 0.0f, 0.0f, 0.0f));
	}
}

void kernel rscActivate(read_only image2d_t inputs, read_only image2d_t statesPrev,
	read_only image3d_t hiddenVisibleWeightsPrev, read_only image3d_t hiddenHiddenPrevWeightsPrev,
	write_only image2d_t activations,
	int2 inputDims, int2 dims, float2 dimsToInputDims,
	int receptiveRadius, int recurrentRadius)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 inputCenterPosition = (int2)((position.x + 0.5f) * dimsToInputDims.x + 0.5f, (position.y + 0.5f) * dimsToInputDims.y + 0.5f);

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
		for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
			int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < inputDims.x && inputPosition.y >= 0 && inputPosition.y < inputDims.y) {
				float input = read_imagef(inputs, inputPosition).x;

				float2 weight = read_imagef(hiddenVisibleWeightsPrev, (int4)(position.x, position.y, wi, 0)).xy;

				float delta = input - weight.x;

				sum += delta * delta * weight.y;
			}

			wi++;
		}

	wi = 0;

	for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
		for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float input = read_imagef(statesPrev, inputPosition).x;

				float2 weight = read_imagef(hiddenHiddenPrevWeightsPrev, (int4)(position.x, position.y, wi, 0)).xy;

				float delta = input - weight.x;

				sum += delta * delta * weight.y;
			}

			wi++;
		}

	write_imagef(activations, position, (float4)(-sum));
}

void kernel rscInhibit(read_only image2d_t activations, read_only image3d_t hiddenHiddenWeightsPrev, read_only image2d_t biasesPrev,
	write_only image2d_t states, write_only image2d_t inhibitions,
	int2 dims,
	int inhibitionRadius)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float thisActivation = read_imagef(activations, position).x;

	float sum = 0.0f;

	int wi = 0;

	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float input = read_imagef(activations, inputPosition).x;

				float weight = read_imagef(hiddenHiddenWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				sum += weight * (input > thisActivation ? 1.0f : 0.0f);
			}

			wi++;
		}

	write_imagef(inhibitions, position, (float4)(sum));

	float bias = read_imagef(biasesPrev, position).x;

	float state = (1.0f - sum * sigmoid(bias)) > 0.0f ? 1.0f : 0.0f;

	write_imagef(states, position, (float4)(state));
}

void kernel rscReconstructReceptive(read_only image2d_t states, read_only image3d_t hiddenVisibleWeightsPrev,
	write_only image2d_t reconstruction,
	int2 inputDims, int2 dims, float2 dimsToInputDims, float2 inputDimsToDims,
	int receptiveRadius, int2 reverseReceptiveRadii)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 hiddenCenterPosition = (int2)((position.x + 0.5f) * inputDimsToDims.x + 0.5f, (position.y + 0.5f) * inputDimsToDims.y + 0.5f);

	float sum = 0.0f;
	float div = 0.0f;

	int wi = 0;

	for (int dx = -reverseReceptiveRadii.x; dx <= reverseReceptiveRadii.x; dx++)
		for (int dy = -reverseReceptiveRadii.y; dy <= reverseReceptiveRadii.y; dy++) {
			int2 hiddenPosition = (int2)(hiddenCenterPosition.x + dx, hiddenCenterPosition.y + dy);

			if (hiddenPosition.x >= 0 && hiddenPosition.x < dims.x && hiddenPosition.y >= 0 && hiddenPosition.y < dims.y) {
				// Project back to input
				int2 fieldCenter = (int2)((hiddenPosition.x + 0.5f) * dimsToInputDims.x + 0.5f, (hiddenPosition.y + 0.5f) * dimsToInputDims.y + 0.5f);

				int2 fieldLowerBounds = fieldCenter - receptiveRadius;
				int2 fieldUpperBounds = fieldCenter + receptiveRadius;

				// Check for containment
				if (position.x >= fieldLowerBounds.x && position.x <= fieldUpperBounds.x && position.y >= fieldLowerBounds.y && position.y <= fieldUpperBounds.y) {
					int rdx = position.x - fieldCenter.x;
					int rdy = position.y - fieldCenter.y;

					float state = read_imagef(states, hiddenPosition).x;

					int wi = (receptiveRadius + rdy) + (receptiveRadius + rdx) * (receptiveRadius * 2 + 1);

					float2 weight = read_imagef(hiddenVisibleWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

					sum += state * weight.y * weight.x;
					div += state * weight.y;
				}
			}

			wi++;
		}

	float recon = sum / fmax(div, epsilon);

	write_imagef(reconstruction, position, (float4)(recon));
}

void kernel rscReconstructRecurrent(read_only image2d_t states, read_only image3d_t hiddenHiddenPrevWeightsPrev,
	write_only image2d_t reconstruction,
	int2 dims,
	int recurrentRadius)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float sum = 0.0f;
	float div = 0.0f;

	int wi = 0;

	for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
		for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
			int2 hiddenPosition = (int2)(position.x + dx, position.y + dy);

			if (hiddenPosition.x >= 0 && hiddenPosition.x < dims.x && hiddenPosition.y >= 0 && hiddenPosition.y < dims.y) {
				float state = read_imagef(states, hiddenPosition).x;

				int wi = (recurrentRadius - dy) + (recurrentRadius - dx) * (recurrentRadius * 2 + 1);

				float2 weight = read_imagef(hiddenHiddenPrevWeightsPrev, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				sum += state * weight.y * weight.x;
				div += state * weight.y;
			}

			wi++;
		}

	float recon = sum / fmax(div, epsilon);

	write_imagef(reconstruction, position, (float4)(recon));
}

void kernel rscError(read_only image2d_t inputs, read_only image2d_t reconstruction,
	write_only image2d_t error)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float input = read_imagef(inputs, position).x;
	float recon = read_imagef(reconstruction, position).x;

	write_imagef(error, position, (float4)(input - recon));
}

void kernel rscLearn(read_only image2d_t receptiveErrors, read_only image2d_t recurrentErrors, read_only image2d_t activations, read_only image2d_t inhibitions, read_only image2d_t states,
	read_only image3d_t hiddenVisibleWeightsPrev, read_only image3d_t hiddenHiddenPrevWeightsPrev, read_only image3d_t hiddenHiddenWeightsPrev, read_only image2d_t biasesPrev,
	write_only image3d_t hiddenVisibleWeights, write_only image3d_t hiddenHiddenPrevWeights, write_only image3d_t hiddenHiddenWeights, write_only image2d_t biases,
	int2 inputDims, int2 dims, float2 dimsToInputDims,
	int receptiveRadius, int recurrentRadius, int inhibitionRadius,
	float4 learningRates, float sparsity, float sparsitySquared)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int2 inputCenterPosition = (int2)((position.x + 0.5f) * dimsToInputDims.x + 0.5f, (position.y + 0.5f) * dimsToInputDims.y + 0.5f);

	float state = read_imagef(states, position).x;

	// Only modify weights if state is 1
	if (state > 0.0f) {
		int wi = 0;

		for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
			for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
				int2 inputPosition = (int2)(inputCenterPosition.x + dx, inputCenterPosition.y + dy);

				if (inputPosition.x >= 0 && inputPosition.x < inputDims.x && inputPosition.y >= 0 && inputPosition.y < inputDims.y) {
					float2 weightPrev = read_imagef(hiddenVisibleWeightsPrev, (int4)(position.x, position.y, wi, 0)).xy;

					float receptiveError = read_imagef(receptiveErrors, inputPosition).x;

					float2 weight = (float2)(weightPrev.x + learningRates.x * receptiveError,
						fmax(epsilon, weightPrev.y + learningRates.y * receptiveError * weightPrev.x));

					write_imagef(hiddenVisibleWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
				}

				wi++;
			}

		wi = 0;

		for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
			for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
				int2 inputPosition = (int2)(position.x + dx, position.y + dy);

				if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
					float2 weightPrev = read_imagef(hiddenHiddenPrevWeightsPrev, (int4)(position.x, position.y, wi, 0)).xy;

					float receptiveError = read_imagef(recurrentErrors, inputPosition).x;

					float2 weight = (float2)(weightPrev.x + learningRates.x * receptiveError,
						fmax(epsilon, weightPrev.y + learningRates.y * receptiveError * weightPrev.x));

					write_imagef(hiddenHiddenPrevWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
				}

				wi++;
			}
	}

	float thisActivation = read_imagef(activations, position).x;
	float thisInhibition = read_imagef(inhibitions, position).x;

	int wi = 0;

	// Inhibitory connections
	for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int2 inputPosition = (int2)(position.x + dx, position.y + dy);

			if (inputPosition.x >= 0 && inputPosition.x < dims.x && inputPosition.y >= 0 && inputPosition.y < dims.y) {
				float input = read_imagef(activations, inputPosition).x;

				float weightPrev = read_imagef(hiddenHiddenWeightsPrev, (int4)(position.x, position.y, wi, 0)).x;

				float weight = fmax(0.0f, weightPrev + learningRates.z * state * (input < thisActivation ? 1.0f : 0.0f) - sparsitySquared * thisInhibition);

				write_imagef(hiddenHiddenWeights, (int4)(position.x, position.y, wi, 0), (float4)(weight));
			}

			wi++;
		}

	// Bias
	float biasPrev = read_imagef(biasesPrev, position).x;

	float bias = learningRates.w * (state - sparsity);

	write_imagef(biases, position, (float4)(bias));
}