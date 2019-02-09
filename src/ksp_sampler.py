import shutil
from arctic import Arctic

from lib import *
from sampler import Sampler

class KSPSampler(Sampler):

	def __init__(self, game, 
		window_episode=None, noise_amplitude_ratio=None, period_range=None, amplitude_range=None,
		fld=None):

		self.n_var = 1	# price only

		self.window_episode = window_episode
		self.noise_amplitude_ratio = noise_amplitude_ratio
		self.period_range = period_range
		self.amplitude_range = amplitude_range
		self.can_half_period = False
		self.quote = None

		self.attrs = ['title','window_episode', 'noise_amplitude_ratio', 'period_range', 'amplitude_range', 'can_half_period']

		param_str = str((
			self.noise_amplitude_ratio, self.period_range, self.amplitude_range
			))
		if game == 'single':
			self.sample = self.__sample_single_sin
			self.title = 'SingleSin'+param_str
		elif game == 'concat':
			self.sample = self.__sample_concat_sin
			self.title = 'ConcatSin'+param_str
		elif game == 'concat_half':
			self.can_half_period = True
			self.sample = self.__sample_concat_sin
			self.title = 'ConcatHalfSin'+param_str
		elif game == 'concat_half_base':
			self.can_half_period = True
			self.sample = self.__sample_concat_sin_w_base
			self.title = 'ConcatHalfSin+Base'+param_str
			self.base_period_range = (int(2*self.period_range[1]), 4*self.period_range[1])
			self.base_amplitude_range = (20,80)
		elif game == 'load':
			self.load_db(fld)
		else:
			raise ValueError


	def build_db(self, n_episodes, fld):
		"""
		db最后的结构是一个数组，每一个元素是一个数据序列+表示该序列数据特征的字符串
		:param n_episodes:
		:param fld:
		:return:
		"""
		store = Arctic('localhost')
		library = store['EOD_PASSTHROUGH']
		self.quote = library.read("sz002024")['ac']
		'''
		根据n_episodes和self.window_episode来slice行情数据，取整数个window_episode
		'''
		episode = len(self.quote.index) // self.window_episode
		if(n_episodes > episode):
			n_episodes = episode
		self.quote = self.quote[-n_episodes*self.window_episode:]

		db = []
		for i in range(n_episodes):
			#prices, title = self.sample()

			prices = []
			p = []

			part = self.quote[i*self.window_episode:((i+1)*self.window_episode)]
			idx = part.index[0].strftime('%Y-%m-%d') + "~" + part.index[-1].strftime('%Y-%m-%d')
			'''
			while True:
				p = np.append(p, self.quote(full_episode=False)[0])
				if len(p) > self.window_episode:
					break
			base, base_title = self.__rand_sin(
				period_range=self.base_period_range,
				amplitude_range=self.base_amplitude_range,
				noise_amplitude_ratio=0.,
				full_episode=True)
			'''
			# p = [x for x in part.values]
			prices.append(np.array(part.values))
			# return np.array(prices).T,


			db.append((np.array(prices).T, '[%i]_'%i+idx))

		print(db)

		if os.path.exists(fld):
			shutil.rmtree(fld)
		os.makedirs(fld)
		# os.makedirs()	# don't overwrite existing fld
		pickle.dump(db, open(os.path.join(fld, 'db.pickle'),'wb'))
		param = {'n_episodes':n_episodes}
		for k in self.attrs:
			param[k] = getattr(self, k)
		json.dump(param, open(os.path.join(fld, 'param.json'),'w'))

	def __rand_sin(self, 
		period_range=None, amplitude_range=None, noise_amplitude_ratio=None, full_episode=False):

		if period_range is None:
			period_range = self.period_range
		if amplitude_range is None:
			amplitude_range = self.amplitude_range
		if noise_amplitude_ratio is None:
			noise_amplitude_ratio = self.noise_amplitude_ratio

		period = random.randrange(period_range[0], period_range[1])
		amplitude = random.randrange(amplitude_range[0], amplitude_range[1])
		noise = noise_amplitude_ratio * amplitude

		if full_episode:
			length = self.window_episode
		else:
			if self.can_half_period:
				length = int(random.randrange(1,4) * 0.5 * period)
			else:
				length = period

		p = 100. + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
		p += np.random.random(p.shape) * noise

		return p, '100+%isin((2pi/%i)t)+%ie'%(amplitude, period, noise)


	def __sample_concat_sin(self):
		prices = []
		p = []
		while True:
			p = np.append(p, self.__rand_sin(full_episode=False)[0])
			if len(p) > self.window_episode:
				break
		prices.append(p[:self.window_episode])
		return np.array(prices).T, 'concat sin'

	def __sample_concat_sin_w_base(self):
		prices = []
		p = []



		while True:
			p = np.append(p, self.quote(full_episode=False)[0])
			if len(p) > self.window_episode:
				break
		base, base_title = self.__rand_sin(
			period_range=self.base_period_range, 
			amplitude_range=self.base_amplitude_range, 
			noise_amplitude_ratio=0., 
			full_episode=True)
		prices.append(p[:self.window_episode] + base)
		return np.array(prices).T, 'concat sin + base: '+base_title
			
	def __sample_single_sin(self):
		prices = []
		funcs = []
		p, func = self.__rand_sin(full_episode=True)
		prices.append(p)
		funcs.append(func)
		return np.array(prices).T, str(funcs)





def t_KSPSampler():

	window_episode = 180
	window_state = 40
	noise_amplitude_ratio = 0.5
	period_range = (10,40)
	amplitude_range = (5,80)
	game = 'concat_half_base'
	instruments = ['fake']

	sampler = KSPSampler(game,
		window_episode, noise_amplitude_ratio, period_range, amplitude_range)
	n_episodes = 100
	"""
	for i in range(100):
		plt.plot(sampler.sample(instruments)[0])
		plt.show()
		"""
	fld = os.path.join('data','KSPSamplerDB',game+'_B')
	sampler.build_db(n_episodes, fld)



if __name__ == '__main__':
	#scan_match()
	t_KSPSampler()
	#p = [1,2,3,2,1,2,3]
	#print find_ideal(p)
	# t_PairSampler()
