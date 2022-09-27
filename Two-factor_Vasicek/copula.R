temperature <- unlist(input[,1])
emissions <- unlist(input[,2])
interests20 <- unlist(input[,3])
interests40 <- unlist(input[,4])
interests60 <- unlist(input[,5])

y20 <- data.frame(interests40, temperature, emissions)

cop_model <- frankCopula(dim = 3)
rotGcop <- rotCopula(frankCopula(dim = 3), flip=c(FALSE,TRUE,FALSE))

m <- pobs(as.matrix(y20))

f2 <- fitCopula(rotGcop, data = m)
coef(f2)
tau_copula(coef(f2), 'frank')

# fit <- fitCopula(cop_model, m, method = 'ml')
# coef(fit)
# tau_copula(coef(fit), 'frank')

#r = sin (.5 πτ)
#0.356503311 (gumbel 20), 0.344099 (gumbel 40), 0.34738516 (gumbel 60)
#0.403918444 - Frank 20
#0.389898324 - Frank 40
#0.390899926- Frank 60

temp_em <- data.frame(emissions,temperature)
copula <- frankCopula(dim = 2)

u <- pobs(as.matrix(temp_em))
fit <- fitCopula(copula, u, method = 'mpl')
coef(fit)
tau(frankCopula(param = coef(fit)))

# Estimate x  gamma distribution parameters and visually compare simulated vs observed data
x_mean <- mean(temperature)
x_var <- var(temperature)
x_rate <- x_mean / x_var
x_shape <- ( (x_mean)^2 ) / x_var


# Estimate y gamma distribution parameters and visually compare simulated vs observed data
y_mean <- mean(emissions)
y_var <- var(emissions)
y_rate <- y_mean / y_var
y_shape <- ( (y_mean)^2 ) / y_var

my_dist <- mvdc(frankCopula(param = -5.877694, dim = 2), margins = c("gamma","gamma"), paramMargins = list(list(shape = x_shape, rate = x_rate), list(shape = y_shape, rate = y_rate)))
v <- rMvdc(1827, my_dist)
