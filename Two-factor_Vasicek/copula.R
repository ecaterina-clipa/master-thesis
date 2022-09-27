#Upload the data from file input_copula.xlsx
temperature <- unlist(input[,1])
emissions <- unlist(input[,2])
interests20 <- unlist(input[,3])
interests40 <- unlist(input[,4])
interests60 <- unlist(input[,5])

y20 <- data.frame(interests20, temperature, emissions)
y40 <- data.frame(interests40, temperature, emissions)
y60 <- data.frame(interests60, temperature, emissions)

cop_model <- frankCopula(dim = 3)
rotGcop <- rotCopula(frankCopula(dim = 3), flip=c(FALSE,TRUE,FALSE))

m20 <- pobs(as.matrix(y20))
m40 <- pobs(as.matrix(y40))
m60 <- pobs(as.matrix(y60))

f20 <- fitCopula(rotGcop, data = m20)
coef(f20)
tau_copula(coef(f20), 'frank')

f40 <- fitCopula(rotGcop, data = m40)
coef(f40)
tau_copula(coef(f40), 'frank')

f60 <- fitCopula(rotGcop, data = m60)
coef(f60)
tau_copula(coef(f60), 'frank')

#r = sin (.5 πτ)

#0.403918444 - Frank 20
#0.389898324 - Frank 40
#0.390899926- Frank 60