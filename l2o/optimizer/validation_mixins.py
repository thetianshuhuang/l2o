

class ValidationMixin:

	def validate(
		self, problems, optimizer, unroll_len=lambda: 20,
		unroll_weights=weights_mean, teachers=[], epochs=1, repeat=1, persistent=False)
